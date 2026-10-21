"""A landed procedure is listed by its structure (rev 2): its landing runs nothing to decide that.

Each world is built by the code that builds it in production: the capture step (``captures.capture``) on each
recorded episode's staged ``procedure.py``, with the episode's signals; ``land_capture`` in the writer's box;
``lifecycle.next_records`` with the store's landings; and the exports that ``memory.match`` and the librarian read
(``dispatch.json``, ``find.json``, the catalogue). Nothing here sets a field of a record.

* a landing records ``recognition`` as ``{"ran": false, "why"}``, and the procedure is listed, however loose its
  ``applies`` (before rev 2 its applies ran on the other recorded requests to decide that);
* the request every reader is handed carries the JSON values the text contains, in order, as the host's
  ``current_request`` gives it.

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
    """The procedure ids the librarian's catalogue lists (``dispatch.json``), and those the structure
    matcher reads (``find.json``'s ``procedures``)."""
    lib = discover(box)
    dispatch = lx.dispatch_data(lib, records=records, captures=captures.export_files(store))
    find = lx.find_data(box, lib, None, None, records, lx.saved_against_of(records))
    return [r["id"] for r in dispatch["procedures"]], [r["id"] for r in find.get("procedures", [])]


@needs_box
@pytest.mark.timeout(900)
def test_a_loose_applies_is_listed_by_its_structure_and_its_landing_records_that_nothing_ran(tmp_path):
    # other work recorded beside it, where the loose applies would hold and its holds fail: never run now
    other = _product_work("e2", [[2, 2], [1, 5, 1]])
    got, records, box = _world(tmp_path, LOOSE, [(other, PRODUCT, "pass")])
    assert got["landing"]["recognition"] == recognition.NOT_RUN
    assert records[PROC]["recognition"] == recognition.NOT_RUN
    assert recognition.LISTED_BY_STRUCTURE in got["reply"]
    dispatched, found = _listed_by_match(box, records, tmp_path / "store")
    assert PROC in dispatched and PROC in found and f"capture:{got['cid']}" in dispatched
    exported = json.loads(captures.export_files(tmp_path / "store")[f".memory/captures/{got['cid']}/record.json"])
    assert "listing" not in exported


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
    assert REQUEST_CLASS in lh._CAPTURE_RUNNER  # rev 2: the recogniser's runner is gone; memory.run's child remains

    def body(fn) -> str:
        src = inspect.getsource(fn)
        return src[src.index("    values: list = []") :]

    assert body(lh._json_values) == body(worker_child.json_values)
    for text in ("", "plain", 'a {"x": [1, {"y": 2}]} b [1, 2] c {bad', "[[1, 2], [3]]\n```json\n{}\n```"):
        assert lh._json_values(text) == worker_child.json_values(text)


def test_the_listing_rule_lists_every_procedure():
    assert recognition.listing(None) == (True, "")
    assert recognition.listing(recognition.NOT_RUN) == (True, recognition.LISTED_BY_STRUCTURE)
    # counts an earlier landing test recorded are not used
    for old in ({"tested": 3, "false_holds": 1}, {"tested": 2, "reads": 1, "fails": 2}, {"error": "load: ImportError"}):
        assert recognition.listing(old) == (True, recognition.EARLIER_TEST_UNUSED)
