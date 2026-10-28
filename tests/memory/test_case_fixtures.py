"""Case fixtures: an input with the output the episode knew for it, re-derived from the recording.

A case file is made by the fixture tool (``fixture(..., expected=...)``): its input is recorded bytes, its expected
output is what the request or the environment showed the actor (a message, including one received after answering,
or an action's response), or the value the episode handed back with the whole request it answered. The file names
its sources, so the gate re-derives it from the recording and a case can never be invented. Keyless and pure: no box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.memory.test_capture_end_to_end import recorded_episode
from tests.memory.test_episodes import _ep
from unify.memory import fixtures as fx
from unify.memory import writer_tools as wt
from unify.memory.episodes import Action

DATA = "memory/pkg/tests/data"


def _no_blob(sha: str) -> bytes:
    raise KeyError(sha)


def _cut(text: str, part: str) -> list[int]:
    """The byte slice of the first *part* in *text*."""
    raw, want = text.encode("utf-8"), part.encode("utf-8")
    start = raw.index(want)
    return [start, start + len(want)]


#: A text request with one worked example, then the instance to answer; after the answer, the environment's reply
#: shows the value it expected.
WORKED = (
    "Turn each name into its initials.\n"
    "Example: Ada Lovelace -> A.L.\n"
    "Now: Grace Brewster Hopper\n"
)
VERDICT = 'Not quite: the expected reply was "G.B.H."'


def worked_episode(eid: str = "w1"):
    return _ep(
        episode_id=eid,
        regime="none",
        request=[WORKED, VERDICT],
        transcript=[{"role": "user", "content": WORKED}],
        cells=[],
        actions=[
            Action(-1, "dialogue:user", "answer", ["G.H."], {}, VERDICT, "ok", "unknown", kind="dialogue"),
            Action(-1, "lookup", "initials", [], {"name": "Ada Lovelace"}, "A.L.", "ok", "read"),
        ],
    )


# --- case files -------------------------------------------------------------------------------------------------


def test_a_shown_case_is_an_input_and_its_output_from_the_request_and_re_derives():
    ep = worked_episode()
    dest = f"{DATA}/initials.example.case.json"
    data, entry = fx.make_case(
        ep,
        {"source": "request:0", "slice": _cut(WORKED, "Ada Lovelace")},
        {"source": "request:0", "slice": _cut(WORKED, "A.L.")},
        dest,
        blob=_no_blob,
    )
    doc = json.loads(data)
    assert (doc["input"], doc["expected"], doc["kind"]) == ("Ada Lovelace", "A.L.", "shown")
    assert entry["kind"] == "case" and entry["case"] == "shown" and entry["episode"] == "w1"
    assert fx.verify(dest, data, entry, {"w1": ep}.get, _no_blob) is None
    case, problem = fx.read_case(dest, data, {"w1": ep}.get, _no_blob)
    assert problem is None and case == {"input": "Ada Lovelace", "expected": "A.L.", "kind": "shown", "episode": "w1"}


def test_an_observation_received_after_answering_may_show_the_expected_output():
    ep = worked_episode()
    data, _ = fx.make_case(
        ep,
        {"source": "request:0", "slice": _cut(WORKED, "Grace Brewster Hopper")},
        {"source": "observation:0", "slice": _cut(VERDICT, "G.B.H.")},
        f"{DATA}/initials.verdict.case.json",
        blob=_no_blob,
    )
    doc = json.loads(data)
    assert (doc["input"], doc["expected"]) == ("Grace Brewster Hopper", "G.B.H.")
    assert doc["source"]["expected"]["source"] == "request:1"


def test_an_api_call_shown_with_its_result_is_a_case():
    ep = worked_episode()
    data, entry = fx.make_case(ep, "args:1", "1", f"{DATA}/initials.call.case.json", blob=_no_blob)
    doc = json.loads(data)
    assert doc["input"] == json.dumps({"args": [], "kwargs": {"name": "Ada Lovelace"}}, sort_keys=True)
    assert doc["expected"] == "A.L." and entry["trust"] == 2


def test_the_source_case_is_the_whole_request_with_the_value_handed_back():
    ep = recorded_episode("e7", rows=((1, 2), (3, 4)))
    data, entry = fx.make_case(ep, "request:0", fx.ANSWER, f"{DATA}/sums.source.case.json", blob=_no_blob)
    doc = json.loads(data)
    assert doc["input"] == ep.request[0] and doc["expected"] == {"sums": [3, 7]} and doc["kind"] == "source"
    assert fx.verify(f"{DATA}/sums.source.case.json", data, entry, {"e7": ep}.get, _no_blob) is None


@pytest.mark.parametrize(
    ("given", "expected", "why"),
    [
        ({"source": "request:0", "slice": [0, 10]}, fx.ANSWER, "pairs only with the whole request"),
        ("request:1", fx.ANSWER, "pairs only with the whole request"),
        ("request:0", "args:0", "not something the request or the environment showed"),
        ("request:0", "cell:0", "not something the request or the environment showed"),
        (fx.ANSWER, "request:0", "answer names an output"),
    ],
)
def test_an_output_the_actor_was_not_shown_is_refused(given, expected, why):
    ep = worked_episode()
    with pytest.raises(fx.FixtureError, match=why):
        fx.make_case(ep, given, expected, f"{DATA}/x.case.json", blob=_no_blob)


def test_an_edited_or_invented_case_does_not_re_derive():
    ep = worked_episode()
    dest = f"{DATA}/initials.example.case.json"
    data, entry = fx.make_case(
        ep,
        {"source": "request:0", "slice": _cut(WORKED, "Ada Lovelace")},
        {"source": "request:0", "slice": _cut(WORKED, "A.L.")},
        dest,
        blob=_no_blob,
    )
    edited = data.replace(b'"A.L."', b'"A.B."', 1)
    case, problem = fx.read_case(dest, edited, {"w1": ep}.get, _no_blob)
    assert case is None and "differs from the case" in problem
    # a case written by hand: its sources name a slice that does not hold its values
    doc = json.loads(data)
    doc["input"] = "Alan Turing"
    case, problem = fx.read_case(dest, fx.case_bytes(doc), {"w1": ep}.get, _no_blob)
    assert case is None and problem
    # not a case file's value at all
    case, problem = fx.read_case(dest, b'{"input": "x", "expected": "y"}', {"w1": ep}.get, _no_blob)
    assert case is None and "made with fixture(..., expected=...)" in problem


def test_the_fixture_tool_writes_a_case_file_that_names_its_sources_and_the_ledger_derives(tmp_path: Path):
    ep = worked_episode()
    box = tmp_path / "box"
    box.mkdir()
    dest = f"{DATA}/initials.example.case.json"
    reply = wt.fixture(
        {
            "episode": "w1",
            "source": "request:0",
            "slice": _cut(WORKED, "Ada Lovelace"),
            "expected": {"source": "request:0", "slice": _cut(WORKED, "A.L.")},
            "dest": dest,
        },
        {"w1": ep}.get,
        box,
        _no_blob,
        ["w1"],
    )
    assert reply.status == "ok", reply.text
    assert "a shown case of w1" in reply.text
    assert not (box / fx.LEDGER_FILE).exists() or dest not in json.loads((box / fx.LEDGER_FILE).read_text())
    ledger = fx.derive_ledger(box)
    assert ledger[dest]["kind"] == "case" and ledger[dest]["case"] == "shown"
    data = (box / dest).read_bytes()
    assert fx.verify(dest, data, ledger[dest], {"w1": ep}.get, _no_blob) is None
    refused = wt.fixture(
        {"episode": "w1", "source": "request:0", "expected": "answer", "dest": f"{DATA}/x.case.json", "append": True},
        {"w1": ep}.get,
        box,
        _no_blob,
        ["w1"],
    )
    assert refused.status != "ok" and "no append and no parser" in refused.text
    wrong_name = wt.fixture(
        {"episode": "w1", "source": "request:0", "expected": "answer", "dest": f"{DATA}/x.json"},
        {"w1": ep}.get,
        box,
        _no_blob,
        ["w1"],
    )
    assert wrong_name.status != "ok" and fx.CASE_SUFFIX in wrong_name.text
