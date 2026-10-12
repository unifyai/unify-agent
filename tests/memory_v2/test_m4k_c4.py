"""m4k-c4 (MAIN 12 Oct; 0f's rehearsal-3 LOW read 50352fbb: all 18 tool errors were ``read_episode`` with part
"parts", refused with a KeyError): asking for an episode's parts answers with the list (each part's size, which are
required and which are read), shows no content and credits nothing."""

from unify.memory_v2 import batch_map as bm
from unify.memory_v2 import fixtures as fx
from unify.memory_v2 import sol_pass, views
from tests.memory_v2.test_sol_pass import _call
from tests.memory_v2.test_sol_v21_tools import EPS, _pass


def _cov(eid):
    row = bm.build_batch_map(EPS.__getitem__, [eid])["episodes"][0]
    req = {eid: row["required_parts"]}
    sizes = {(eid, p): len(bm.part_text(EPS[eid], p).encode()) for p in req[eid]}
    return views.Coverage(req, sizes)


def test_the_parts_list_names_every_part_with_its_size_and_state():
    cov = _cov("e1")
    text = sol_pass._parts_list(EPS["e1"], "e1", cov)
    lines = text.splitlines()
    assert lines[0] == "e1: 4 part(s), 3 required"
    size = len(bm.part_text(EPS["e1"], "request").encode())
    assert f"- request: {size} bytes, required, not read" in lines
    assert any(
        ln.startswith("- observation:1: ") and "identical to observation:0" in ln
        for ln in lines
    )
    assert any(
        ln.startswith("- cell:0: ") and ln.endswith("required, not read")
        for ln in lines
    )
    assert lines[-1].startswith("Read parts with parts=[...] (8 per call)")
    assert "def f()" not in text  # no content
    cov.credit("e1", "request", 0, size)
    assert f"- request: {size} bytes, required, read" in sol_pass._parts_list(
        EPS["e1"],
        "e1",
        cov,
    )


def test_read_episode_parts_answers_with_the_list_and_credits_nothing(tmp_path):
    turns = [
        _call("p1", "read_episode", {"episode": "e1", "part": "parts", "offset": 0}),
        _call("r", "read_episode", {"episode": "e1", "part": "request"}),
        _call("p2", "read_episode", {"episode": "e1", "part": "parts"}),
    ]
    out, model = _pass(tmp_path, turns, eids=("e1",), max_calls=5)
    first, second = model.outputs["p1"], model.outputs["p2"]
    assert (
        not first.startswith("refused")
        and first.splitlines()[0] == "e1: 4 part(s), 3 required"
    )
    size = len(bm.part_text(EPS["e1"], "request").encode())
    assert f"- request: {size} bytes, required, not read" in first
    assert f"- request: {size} bytes, required, read" in second
    assert (
        out.coverage["parts_read"] == 1
    )  # only the request: the lists credited nothing


def test_the_tool_description_names_the_parts_list():
    desc = next(
        t
        for t in sol_pass.sol_tools(v21=True)
        if t["function"]["name"] == "read_episode"
    )
    assert "'parts' lists the episode's parts" in desc["function"]["description"]


# --- rehearsal 3, final read (0f 57e57345): a slice past the record's end, and an unknown part ---------------------


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
    # the recorded (clamped) slice re-derives the same bytes for the gate
    assert fx.verify(dest, data, entry, EPS.get, lambda s: b"") is None
    # appended lines record the clamped slice too
    line, assembled = fx.make(
        ep,
        "request:0",
        "memory/arc/tests/data/req.inputs.jsonl",
        blob=lambda s: b"",
        cut=[1, 50],
        append=True,
        notes=[],
    )
    assert assembled["lines"][0]["slice"] == [1, 3] and b'"slice": [1, 3]' in line
    # a slice that starts past the end, or is empty, is still refused: nothing exists to copy
    for bad in ([3, 9], [0, 0], [-1, 2]):
        try:
            fx.make(ep, "request:0", dest, blob=lambda s: b"", cut=bad, notes=[])
        except fx.FixtureError:
            continue
        raise AssertionError(f"{bad} was accepted")


def test_the_fixture_tool_reports_the_truncation(tmp_path):
    made: dict = {}
    out = sol_pass._fixture_call(
        {
            "episode": "e1",
            "action": "request:0",
            "dest": "memory/arc/tests/data/req.txt",
            "slice": [0, 64],
        },
        EPS,
        tmp_path,
        lambda s: b"",
        made,
    )
    assert out.startswith("ok: wrote memory/arc/tests/data/req.txt (3 bytes")
    assert out.endswith(
        "; truncated: slice end 64 is past the 3 recorded bytes; copied [0, 3]",
    )
    assert (tmp_path / "memory/arc/tests/data/req.txt").read_bytes() == b"req"


def test_an_unknown_part_answers_with_the_episode_parts(tmp_path):
    turns = [
        _call("one", "read_episode", {"episode": "e1", "part": "cell:2"}),
        _call(
            "many",
            "read_episode",
            {"episode": "e1", "parts": ["request", "observation:9"]},
        ),
    ]
    out, model = _pass(tmp_path, turns, eids=("e1",), max_calls=4)
    assert model.outputs["one"].startswith(
        "no part 'cell:2' in e1; nothing was read. Its parts:\ne1: 4 part(s)",
    )
    assert model.outputs["many"].startswith(
        "no part 'observation:9' in e1; nothing was read. Its parts:",
    )
    assert "- cell:0: " in model.outputs["one"]
    assert out.coverage["parts_read"] == 0  # nothing shown, nothing credited
