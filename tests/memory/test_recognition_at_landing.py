"""A landed procedure's ``applies`` is tested, when it lands, on the other requests the capture store records.

Each world is built by the code that builds it in production: the capture step (``captures.capture``) on each
recorded episode's staged ``procedure.py``, with the episode's signals; ``land_capture`` in the writer's box, which
runs the landed reader, ``applies``, procedure and ``holds`` confined on the other recorded requests and records the
counts with the landing; ``lifecycle.next_records`` with the store's landings; and the exports that ``memory.match``
and the librarian read (``dispatch.json``, ``find.json``, the catalogue). Nothing here sets a field of a record.

* a loose ``applies`` that holds on a recorded request of other work, where the procedure's own ``holds`` fails, is a
  false hold, and the procedure is not listed;
* an ``applies`` that holds where the procedure reproduces the value a successful episode handed back is a true hold,
  and the procedure is listed;
* reproducing the value of an episode recorded as failed is unknown, never a true hold;
* a landing with no other recorded request is listed, untested;
* a reader that fails on a recorded variant of its own request (a request of the same work recognised by content, or
  its own request with blank lines around it or a line of prose after it) is brittle, and its procedure is not listed;
  a reader that reads the request's data (``request.data``, the JSON values it carries) survives the same variants;
* the request every reader is handed carries that data, as the host's ``current_request`` gives it: the JSON values
  the text contains, in order.

Keyless; the capture step and the landing run code confined, so they need bubblewrap and prlimit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.memory.test_episodes import _ep
from tests.memory.test_tool_channel_box import needs_box
from unify.memory import captures
from unify.memory import land_capture
from unify.memory import library_export as lx
from unify.memory import lifecycle
from unify.memory import recognition
from unify.memory.episodes import Action, Cell
from unify.memory.layout import discover
from unify.memory.signals import Signal

PROC = "memory.readings.procedures:row_sums"
READER = "memory.readings.readers:read_row_sums"

#: Sums each row; its applies holds on any request with rows (loose), and its holds checks the request's own worked
#: case: summing the case's numbers gives the stated result.
LOOSE = '''"""Sum each row of numbers read from the request."""


def read_input(text):
    """The rows (lines that start with row:) and the worked cases (lines that start with Worked case:)."""
    rows, cases = [], []
    for line in text.splitlines():
        if line.startswith("row:"):
            rows.append([int(x) for x in line[len("row:"):].split()])
        elif line.startswith("Worked case:"):
            left, right = line[len("Worked case:"):].split("->")
            cases.append([[int(x) for x in left.split()], int(right)])
    if not rows:
        raise ValueError("no rows")
    return {"rows": rows, "cases": cases}


def run(value):
    """The sum of each row."""
    return {"sums": [sum(row) for row in value["rows"]]}


def applies(value):
    """Any request with rows."""
    return bool(value["rows"])


def holds(value, result):
    """One sum per row, and summing each worked case gives its stated result."""
    return len(result["sums"]) == len(value["rows"]) and all(sum(c) == want for c, want in value["cases"])
'''
#: The same procedure stating no holds: whether it is right on another request rests on what that episode handed back.
NO_HOLDS = LOOSE[: LOOSE.index("\n\ndef holds(")] + "\n"
#: Another attempt's procedure (other work: the product of each row), captured from the other episodes.
PRODUCT = '''"""Multiply each row of numbers read from the request."""


def read_input(text):
    """The rows: lines that start with row:."""
    return [[int(x) for x in line[len("row:"):].split()] for line in text.splitlines() if line.startswith("row:")]


def run(value):
    """The product of each row."""
    out = []
    for row in value:
        p = 1
        for x in row:
            p *= x
        out.append(p)
    return {"sums": out}
'''
#: The same procedure with a reader that reads by line position: the rows between the first line and the last two,
#: the worked case on the line before the last.
BRITTLE = LOOSE.replace(
    LOOSE[LOOSE.index("def read_input(") : LOOSE.index("\n\ndef run(")],
    '''def read_input(text):
    """The rows: the lines between the first line and the last two; the worked case: the line before the last."""
    lines = text.splitlines()
    rows = [[int(x) for x in line[len("row:"):].split()] for line in lines[1:-2]]
    left, right = lines[-2][len("Worked case:"):].split("->")
    return {"rows": rows, "cases": [[[int(x) for x in left.split()], int(right)]]}''',
)
SUMMARY = "## Asked\nCombine each row of numbers.\n## Approach\n- Read the rows.\n## Outcome\nReplied.\n"


def _request(rows, case, op: str, tail: str = "") -> str:
    lines = "\n".join("row: " + " ".join(str(x) for x in r) for r in rows)
    return (
        f"Readings, one row per line:\n{lines}\nWorked case: {' '.join(str(x) for x in case[0])} -> {case[1]}\n"
        f"Reply with the {op} of each row." + tail
    )


def _episode(eid: str, rows, case, op: str, answer, tail: str = ""):
    """A recorded episode: a text request and the dialogue action that handed *answer* back."""
    text = _request(rows, case, op, tail)
    return _ep(
        episode_id=eid,
        request=[text],
        transcript=[{"role": "user", "content": text}],
        cells=[Cell(0, "rows = ...", repr(answer))],
        actions=[Action(-1, "dialogue:user", "answer", [], {"sums": answer}, "Noted.", "ok", "unknown", kind="dialogue")],
    )


def _sum_work(eid: str, rows, tail: str = ""):
    return _episode(eid, rows, ([2, 3], 5), "sum", [sum(r) for r in rows], tail)


def _product_work(eid: str, rows):
    products = []
    for r in rows:
        p = 1
        for x in r:
            p *= x
        products.append(p)
    return _episode(eid, rows, ([2, 3], 6), "product", products)


def _checker(eid: str, label: str) -> Signal:
    return Signal(f"s-{eid}", eid, "checker", label, "t", regime="dense", visible_to_actor=True)


def _capture(root: Path, ep, source: str, verdict: str | None) -> dict:
    """The capture step on *ep* with *source* staged, and the checker's visible verdict when given."""
    staging = root / "staging" / ep.episode_id
    staging.mkdir(parents=True)
    (staging / captures.PROCEDURE_FILE).write_text(source, encoding="utf-8")
    (staging / captures.SUMMARY_FILE).write_text(SUMMARY, encoding="utf-8")
    signals = [_checker(ep.episode_id, verdict)] if verdict else []
    return captures.capture(staging, ep, root / "store", signals=signals, checker_visible=True)


def _world(
    tmp_path: Path, source: str, others: list[tuple], before: list[tuple] = (), own=None,
) -> tuple[dict, dict, Path]:
    """Capture each of *before* (episode, staged source, verdict), then *own* (default: e1, the sum work, accepted)
    with *source*, then each of *others*, then land its capture. Returns (the landing reply's text, the landing and
    the capture id; the item records; the box)."""
    own = own if own is not None else _sum_work("e1", [[1, 2, 2], [3, 4, 5]])
    episodes = {own.episode_id: own}
    for ep, staged, verdict in before:
        episodes[ep.episode_id] = ep
        _capture(tmp_path, ep, staged, verdict)
    row = _capture(tmp_path, own, source, "pass")
    assert row["safety"] == "ok", row
    for ep, staged, verdict in others:
        episodes[ep.episode_id] = ep
        _capture(tmp_path, ep, staged, verdict)
    box = tmp_path / "box"
    box.mkdir()
    reply = land_capture.land(
        {"capture_id": row["capture_id"], "package": "readings", "name": "row_sums"},
        store=tmp_path / "store",
        box=box,
        episode=episodes.get,
        blob=_no_blob,
        batch=sorted(episodes),
    )
    assert reply.status == "ok", reply.text
    landing = captures.landings(tmp_path / "store")[row["capture_id"]]
    return {"reply": reply.text, "landing": landing, "cid": row["capture_id"]}, _records(box, tmp_path / "store"), box


def _no_blob(sha: str) -> bytes:
    raise KeyError(sha)


def _records(box: Path, store: Path) -> dict:
    """The item records of the box's library, from the store's landings (``lifecycle.next_records``)."""
    lib = discover(box)
    items = {f.item_id: "function" for f in lib.functions}
    sha = "a" * 40
    recs, _ = lifecycle.next_records(
        items=items,
        prev={},
        facts=[],
        since={i: {sha} for i in items},
        changed_at={i: sha for i in items},
        functions={f.item_id: f.module for f in lib.functions},
        graph={f.module: set() for f in lib.functions},
        note_uses={},
        landed=captures.landed_items(store),
    )
    return recs


def _listed_by_match(box: Path, records: dict, store: Path) -> tuple[list[str], list[str]]:
    """The procedure ids ``memory.match`` and the librarian's facts run (``dispatch.json``), and those the structure
    matcher reads (``find.json``'s ``procedures``)."""
    lib = discover(box)
    dispatch = lx.dispatch_data(lib, records=records, captures=captures.export_files(store))
    find = lx.find_data(box, lib, None, None, records, lx.saved_against_of(records))
    return [r["id"] for r in dispatch["procedures"]], [r["id"] for r in find.get("procedures", [])]


@needs_box
@pytest.mark.timeout(900)
def test_a_loose_applies_that_holds_where_its_holds_fails_is_a_false_hold_and_not_listed(tmp_path):
    # other work recorded beside it: the product of each row, accepted; its worked case says 2 3 -> 6
    other = _product_work("e2", [[2, 2], [1, 5, 1]])
    got, records, box = _world(tmp_path, LOOSE, [(other, PRODUCT, "pass")])
    rec = got["landing"]["recognition"]
    assert (rec["requests"], rec["tested"], rec["true_holds"], rec["false_holds"], rec["unknown"]) == (1, 1, 0, 1, 0)
    assert rec["listed"] is False and rec["why"].startswith("not listed: its applies held on 1 of 1")
    assert records[PROC]["recognition"] == rec
    assert "recognition on the recorded requests: not listed" in got["reply"]
    dispatched, found = _listed_by_match(box, records, tmp_path / "store")
    assert PROC not in dispatched and PROC not in found
    # the captured procedure is the same code: it is not listed either, and its exported record says why
    assert f"capture:{got['cid']}" not in dispatched
    exported = json.loads(captures.export_files(tmp_path / "store")[f".memory/captures/{got['cid']}/record.json"])
    assert exported["listing"] == {"listed": False, "why": rec["why"]}
    # the card and memory.get say why, so the writer sees it at its next pass
    card = lx.catalogue_text(box, records=records, captures=captures.export_files(tmp_path / "store"))
    assert f"Recognition: {rec['why']}" in card
    items = lx.items_data(discover(box), None, {}, True, records)
    assert items["items"][PROC]["recognition"] == rec["why"]


@needs_box
@pytest.mark.timeout(900)
def test_an_applies_that_holds_where_the_procedure_reproduces_a_successful_episode_is_a_true_hold_and_listed(
    tmp_path,
):
    # the same work, visited again with other rows and accepted: the procedure states no holds, so whether it is
    # right there rests on reproducing the value that episode handed back
    later = _sum_work("e2", [[9, 1], [0, 5], [7, 7]])
    got, records, box = _world(tmp_path, NO_HOLDS, [(later, PRODUCT, "pass")])
    rec = got["landing"]["recognition"]
    assert (rec["requests"], rec["tested"], rec["true_holds"], rec["false_holds"], rec["unknown"]) == (1, 1, 1, 0, 0)
    assert rec["listed"] is True and rec["why"].startswith("tested on 1 recorded request(s): 1 true hold(s)")
    # its reader reads every recorded variant of its own request: the three mechanical ones
    assert (rec["variants"], rec["reads"], rec["fails"], rec["failed_on"]) == (3, 3, 0, [])
    dispatched, found = _listed_by_match(box, records, tmp_path / "store")
    assert PROC in dispatched and PROC in found
    assert f"capture:{got['cid']}" in dispatched


@needs_box
@pytest.mark.timeout(900)
def test_reproducing_the_value_of_an_episode_recorded_as_failed_is_unknown_never_a_true_hold(tmp_path):
    # the same work visited again, but the checker rejected what that episode handed back
    rejected = _sum_work("e2", [[9, 1], [0, 5], [7, 7]])
    got, records, box = _world(tmp_path, NO_HOLDS, [(rejected, PRODUCT, "fail")])
    assert recognition.episode_outcomes(tmp_path / "store")["e2"] == "failure"
    rec = got["landing"]["recognition"]
    assert (rec["tested"], rec["true_holds"], rec["false_holds"], rec["unknown"]) == (1, 0, 0, 1)
    # and an episode the store records no outcome for is unknown too
    silent = _sum_work("e3", [[4, 4]])
    got2, _, _ = _world(tmp_path / "silent", NO_HOLDS, [(silent, PRODUCT, None)])
    assert "e3" not in recognition.episode_outcomes(tmp_path / "silent" / "store")
    rec2 = got2["landing"]["recognition"]
    assert (rec2["tested"], rec2["true_holds"], rec2["unknown"]) == (1, 0, 1)
    assert rec["listed"] is True and rec2["listed"] is True


@needs_box
@pytest.mark.timeout(900)
def test_a_landing_with_no_other_recorded_request_is_listed_untested(tmp_path):
    got, records, box = _world(tmp_path, LOOSE, [])
    rec = got["landing"]["recognition"]
    assert (rec["requests"], rec["tested"], rec["false_holds"]) == (0, 0, 0)
    assert rec["listed"] is True
    assert rec["why"] == "untested: no other recorded request to test its applies on"
    dispatched, found = _listed_by_match(box, records, tmp_path / "store")
    assert PROC in dispatched and PROC in found
    card = lx.catalogue_text(box, records=records)
    assert "Recognition: untested: no other recorded request to test its applies on" in card


@needs_box
@pytest.mark.timeout(900)
def test_a_request_the_reader_does_not_parse_is_not_tested_and_applies_that_does_not_hold_is_no_hold(tmp_path):
    # a request with no rows: the reader raises, so it is not tested
    norows = _ep(
        episode_id="e2",
        request=["Nothing to read here."],
        actions=[Action(-1, "dialogue:user", "answer", [], {"sums": []}, "Noted.", "ok", "unknown", kind="dialogue")],
    )
    got, _, _ = _world(tmp_path, LOOSE, [(norows, PRODUCT, "pass")])
    rec = got["landing"]["recognition"]
    assert (rec["requests"], rec["tested"], rec["false_holds"], rec["true_holds"], rec["unknown"]) == (1, 0, 0, 0, 0)
    assert rec["listed"] is True and rec["why"] == "untested: its reader parsed none of the 1 other recorded request(s)"


@needs_box
@pytest.mark.timeout(900)
def test_a_reader_that_fails_on_a_recorded_variant_of_its_own_request_is_brittle_and_not_listed(tmp_path):
    # the same work visited with a line after the request, captured first with a reader that reads by line prefix;
    # e1's capture is recognised as the same work when it is captured (its run on e2's request gives e2's value)
    later = _sum_work("e2", [[9, 1], [0, 5], [7, 7]], tail="\nThanks for checking.")
    got, records, box = _world(tmp_path, BRITTLE, [], before=[(later, LOOSE, "pass")])
    store = tmp_path / "store"
    same = captures.load(store)[got["cid"]]["record"]["same_as"]
    assert len(same) == 1 and "e2" in captures.load(store)[same[0]]["record"]["episodes"]
    rec = got["landing"]["recognition"]
    # e2's request and the three mechanical variants of its own: it reads none of them
    assert (rec["variants"], rec["reads"], rec["fails"]) == (4, 0, 4)
    assert rec["failed_on"] == [
        recognition.RECORDED_VARIANT,
        recognition.LEADING_BLANK,
        recognition.TRAILING_BLANK,
        recognition.TRAILING_PROSE,
    ]
    assert rec["false_holds"] == 0 and rec["listed"] is False
    assert rec["why"].startswith("not listed: its reader failed on 4 of 4 recorded variant(s) of its own request")
    assert records[PROC]["recognition"] == rec
    dispatched, found = _listed_by_match(box, records, store)
    assert PROC not in dispatched and PROC not in found
    card = lx.catalogue_text(box, records=records, captures=captures.export_files(store))
    assert f"Recognition: {rec['why']}" in card


#: A procedure whose reader reads the request's data: the rows of the first JSON value it carries.
DATA_READER = '''"""Sum each row of numbers the request carries as JSON."""

import json


def read_input(request):
    """The rows: the "rows" of the first JSON value the request carries."""
    return request.data[0]["rows"]


def run(value):
    """The sum of each row."""
    return {"sums": [sum(row) for row in value]}


def applies(value):
    """A list of rows."""
    return isinstance(value, list) and all(isinstance(row, list) for row in value)
'''
#: The same procedure with a reader of the request's layout: the JSON on its second line.
LAYOUT_READER = DATA_READER.replace(
    '''    """The rows: the "rows" of the first JSON value the request carries."""
    return request.data[0]["rows"]''',
    '''    """The rows: the "rows" of the JSON object on the request's second line."""
    return json.loads(request.splitlines()[1])["rows"]''',
)


def _json_work(eid: str, text: str, rows):
    return _ep(
        episode_id=eid,
        request=[text],
        transcript=[{"role": "user", "content": text}],
        cells=[Cell(0, "rows = ...", "ok")],
        actions=[
            Action(-1, "dialogue:user", "answer", [], {"sums": [sum(r) for r in rows]}, "Noted.", "ok", "unknown",
                   kind="dialogue"),
        ],
    )


#: The same work twice: the rows as JSON, inside other prose and another layout on the later visit.
OWN_JSON = _json_work("e1", 'Readings for the week:\n{"rows": [[1, 2, 2], [3, 4, 5]]}\nReply with the sum of each row.',
                      [[1, 2, 2], [3, 4, 5]])
LATER_JSON = _json_work(
    "e2",
    "Here are the readings, as before.\nThey came in late this week.\n```json\n"
    '{"rows": [[9, 1], [0, 5], [7, 7]]}\n```\nReply with the sum of each row. Thanks.',
    [[9, 1], [0, 5], [7, 7]],
)


@needs_box
@pytest.mark.timeout(900)
@pytest.mark.parametrize("reader", ["data", "layout"])
def test_a_reader_of_the_requests_data_survives_its_recorded_variants_and_a_reader_of_its_layout_does_not(
    tmp_path, reader,
):
    # e2 is captured first by another attempt's code that reads the data; e1's capture is recognised as the same work
    other = DATA_READER + "\n\n# written again in another attempt\n"
    source = DATA_READER if reader == "data" else LAYOUT_READER
    got, records, box = _world(tmp_path, source, [], before=[(LATER_JSON, other, "pass")], own=OWN_JSON)
    store = tmp_path / "store"
    assert len(captures.load(store)[got["cid"]]["record"]["same_as"]) == 1
    rec = got["landing"]["recognition"]
    assert rec["variants"] == 4
    dispatched, _ = _listed_by_match(box, records, store)
    if reader == "data":
        # the capture step, the landing's fixtures and its recognition test all handed the reader the request's data
        assert (rec["reads"], rec["fails"], rec["failed_on"]) == (4, 0, [])
        # e2: it parsed, its applies held, and it reproduced the value e2 handed back, which was accepted
        assert (rec["tested"], rec["true_holds"], rec["false_holds"]) == (1, 1, 0)
        assert rec["listed"] is True and PROC in dispatched
        test = (box / "memory" / "readings" / "tests" / "test_row_sums.py").read_text()
        assert "read_row_sums(request(text))" in test and "\ndef request(text):" in test
    else:
        assert (rec["reads"], rec["fails"]) == (2, 2)
        assert rec["failed_on"] == [recognition.RECORDED_VARIANT, recognition.LEADING_BLANK]
        assert rec["listed"] is False and "(request.data)" in rec["why"] and PROC not in dispatched


@needs_box
@pytest.mark.timeout(900)
def test_a_landed_reader_of_the_requests_data_passes_its_landed_tests_in_the_gate(tmp_path):
    from tests.memory.gate_records import Landing, build

    built = build(
        tmp_path,
        episodes={"e1": OWN_JSON},
        base={},
        landings=[Landing("e1", DATA_READER, SUMMARY, "readings", "row_sums")],
    )
    # the gate ran the landed test module (its request helper and the reader on the recorded text) and passed it
    assert built.result.passed
    assert "def request(text):" in built.files["memory/readings/tests/test_row_sums.py"]


def test_current_request_gives_the_json_values_a_text_carries_in_order():
    from types import SimpleNamespace

    from unify.actor import tool_handlers as H
    from unify.actor.execution.worker_child import Request

    text = 'Two tables: {"rows": [[1, 2]]} and, after some prose, [[3, 4], [5]] -- and no third { here.'
    want = [{"rows": [[1, 2]]}, [[3, 4], [5]]]
    assert H.request_of(text) == (text, want)
    state = SimpleNamespace(request_text=text, request_data=want)
    assert json.loads(H.current_request(SimpleNamespace(state=state, agent="actor"), {}).text) == {
        "text": text,
        "data": want,
    }
    # exactly the bound request's data, and what every reader is handed with the text
    assert Request(text).data == want
    from unify.memory.request_value import request_data

    assert request_data(text) == want


def test_the_actor_side_runners_hand_readers_the_same_request():
    """The memory package runs readers in the actor's box, where the harness cannot be imported: its copies of the
    request class and of the scan are the harness's, byte for byte and value for value."""
    import inspect

    from unify.actor.execution import worker_child
    from unify.memory import library_helper as lh
    from unify.memory.request_value import REQUEST_CLASS

    assert lh._REQUEST_CLASS == REQUEST_CLASS
    assert REQUEST_CLASS in lh._RECOGNISER_RUNNER and REQUEST_CLASS in lh._CAPTURE_RUNNER

    def body(fn) -> str:
        src = inspect.getsource(fn)
        return src[src.index("    values: list = []") :]

    assert body(lh._json_values) == body(worker_child.json_values)
    for text in ("", "plain", 'a {"x": [1, {"y": 2}]} b [1, 2] c {bad', "[[1, 2], [3]]\n```json\n{}\n```"):
        assert lh._json_values(text) == worker_child.json_values(text)


def test_the_listing_rule_reads_the_counts():
    assert recognition.listing(None) == (True, "")
    assert recognition.listing({"tested": 3, "false_holds": 1})[0] is False
    assert recognition.listing({"tested": 0, "false_holds": 0}) == (
        True,
        "untested: no other recorded request to test its applies on",
    )
    listed, why = recognition.listing({"tested": 2, "true_holds": 1, "unknown": 1, "false_holds": 0})
    assert listed and why == "tested on 2 recorded request(s): 1 true hold(s), 0 false holds, 1 unknown"
    assert recognition.listing({"error": "load: ImportError"})[1].startswith("untested: its test could not be run")
    brittle = recognition.listing({"tested": 2, "reads": 1, "fails": 2, "failed_on": [recognition.TRAILING_PROSE]})
    assert brittle == (
        False,
        "not listed: its reader failed on 2 of 3 recorded variant(s) of its own request (a line of prose after it); "
        "make it read the values the request carries (request.data), not the layout around them",
    )
