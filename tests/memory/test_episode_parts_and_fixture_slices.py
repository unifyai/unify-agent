"""read_episode's table of contents (every part, its size, its fixture source and that source's raw length), unknown
names answered with the valid ones, any recorded episode readable, and fixture slices on the same numbering.
"""

import json

from unify.memory import batch_map as bm
from unify.memory import fixtures as fx
from unify.memory import writer_tools as wt
from tests.memory.test_fixtures import BLOB, EP, _blob
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_library_tools import EPS, _episode, _pass


def test_the_contents_name_every_part_with_size_and_fixture_source_length():
    text = wt.contents(EP, "e1", _blob)
    lines = text.splitlines()
    assert lines[0].startswith("e1: 7 part(s).")
    size = len(bm.part_text(EP, "request").encode())
    assert f"- request: {size} bytes; fixture source: request:0 (18 bytes)" in lines
    obs = next(ln for ln in lines if ln.startswith("- observation:0: "))
    assert obs.endswith("fixture source: request:1 (9 bytes)")
    act2 = next(ln for ln in lines if ln.startswith("- action:2: "))
    assert f"fixture source: 2 ({len(BLOB)} bytes), args:2 (" in act2
    act3 = next(ln for ln in lines if ln.startswith("- action:3: "))
    assert (
        "fixture source: args:3 (" in act3 and " 3 (" not in act3
    )  # no recorded response
    assert "Ada" not in text  # no content


def test_read_episode_without_parts_answers_with_the_contents(tmp_path):
    turns = [
        _call("toc", "read_episode", {"episode": "e1"}),
        _call("p", "read_episode", {"episode": "e1", "part": "parts"}),
        _call("l", "read_episode", {"episode": "e1", "parts": ["parts"]}),
    ]
    out, model = _pass(tmp_path, turns, eids=("e1",), max_calls=5)
    for cid in ("toc", "p", "l"):
        assert model.outputs[cid].startswith("e1: 4 part(s)."), model.outputs[cid]
    assert out.coverage["opened"] == []  # the contents show no part


def test_an_unknown_part_answers_with_the_valid_names(tmp_path):
    turns = [
        _call("one", "read_episode", {"episode": "e1", "part": "cell:2"}),
        _call(
            "many",
            "read_episode",
            {"episode": "e1", "parts": ["request", "observation:9"]},
        ),
    ]
    out, model = _pass(tmp_path, turns, eids=("e1",), max_calls=4)
    assert model.outputs["one"] == (
        "refused: no part 'cell:2' in e1; nothing was shown. Valid: request, observation:0..1, cell:0..0"
    )
    assert model.outputs["many"].startswith(
        "unknown part(s) in e1: 'observation:9'; valid: ",
    )
    assert "Error" not in model.outputs["one"] and "Error" not in model.outputs["many"]


def test_any_recorded_episode_is_readable_and_others_are_refused(tmp_path, monkeypatch):
    other = _episode("old1", request=("an older request",))
    EPS["old1"] = other
    try:
        turns = [
            _call("old", "read_episode", {"episode": "old1", "parts": ["request"]}),
            _call("ov", "overview", {"episode": "old1"}),
            _call("no", "read_episode", {"episode": "never", "parts": ["request"]}),
            _call("path", "read_episode", {"episode": "../e1", "parts": ["request"]}),
        ]
        from tests.memory import test_writer_library_tools as t

        model = t.Turns(turns)
        _, ev, writer = t._writer(tmp_path, model, max_calls=8)
        writer.load = EPS.__getitem__
        monkeypatch.setattr(
            ev,
            "experience_since",
            lambda after: [("old1", 0), ("e1", 0)],
        )
        import asyncio

        out = asyncio.run(
            writer.run(t.PassRequest("incremental", "svc", ["e1"], False), "p1"),
        )
    finally:
        del EPS["old1"]
    assert model.outputs["old"] == '== request ==\n"an older request"'
    assert model.outputs["ov"].startswith("== end ==")
    assert (
        model.outputs["no"]
        == "refused: 'never' is not a recorded episode; this batch: e1"
    )
    assert model.outputs["path"].startswith(
        "refused: '../e1' is not a recorded episode",
    )
    assert out.coverage["opened"] == ["old1"] and out.coverage["unread"] == ["e1"]


def test_observation_alias_names_the_same_bytes_as_request_plus_one():
    a, ea = fx.make(EP, "observation:0", "memory/a/tests/data/o.txt", blob=_blob)
    b, eb = fx.make(EP, "request:1", "memory/a/tests/data/o.txt", blob=_blob)
    assert a == b == b"obs: done" and ea == eb
    assert fx.source_bytes(EP, "observation:0", _blob) == b"obs: done"


def test_a_slice_past_the_record_copies_what_exists_and_says_it_was_truncated():
    ep, dest = EPS["e1"], "memory/arc/tests/data/req.txt"
    notes: list[str] = []
    data, entry = fx.make(
        ep,
        "request:0",
        dest,
        blob=lambda s: b"",
        cut=[0, 100],
        notes=notes,
    )
    assert data == b"req" and entry["slice"] == [0, 3]
    assert notes == [
        "truncated: slice end 100 is past the 3 recorded bytes; copied [0, 3]",
    ]
    assert fx.verify(dest, data, entry, EPS.get, lambda s: b"") is None
    line, assembled = fx.make(
        ep,
        "request:0",
        "memory/arc/tests/data/req.inputs.jsonl",
        blob=lambda s: b"",
        cut=[1, None],
        append=True,
        notes=[],
    )
    assert assembled["lines"][0]["slice"] == [1, 3] and json.loads(line)["source"][
        "slice"
    ] == [1, 3]
    for bad in ([3, 9], [0, 0], [-1, 2]):
        try:
            fx.make(ep, "request:0", dest, blob=lambda s: b"", cut=bad, notes=[])
        except fx.FixtureError:
            continue
        raise AssertionError(f"{bad} was accepted")
