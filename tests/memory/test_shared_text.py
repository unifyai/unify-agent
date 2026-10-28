"""Request text shared by most episodes is shown once, as a marker with its file."""

import asyncio
import json

from unify.memory import batch_map as bm
from unify.memory import shared_text
from unify.memory.evidence import EvidenceStore
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import _episode

PRE = "### Team record\n" + "You may work with other agents. " * 12


def test_a_block_in_most_episodes_becomes_one_marker(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for i in range(4):
        shared_text.update_counts(ev, f"e{i}", PRE + f"\n\nTask {i}: do thing {i}")
    shared_text.update_counts(
        ev,
        "e1",
        PRE + "\n\nTask 1: do thing 1",
    )  # counted once per episode
    sh = shared_text.shared(ev)
    assert PRE in sh.values()  # the block (its long line is a shared unit too)
    marked = shared_text.mark(PRE + "\n\nTask 1: do thing 1", sh)
    bid = next(b for b, t in sh.items() if t == PRE)
    assert marked == shared_text.marker(bid) + "\n\nTask 1: do thing 1"


def test_shared_blocks_edge_cases(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert shared_text.shared(ev) == {}  # nothing seen: no table, nothing shared
    shared_text.update_counts(ev, "e0", PRE)
    assert shared_text.shared(ev) == {}  # one episode: nothing is shared
    shared_text.update_counts(ev, "e1", PRE)
    sh = shared_text.shared(ev)
    assert shared_text.mark(PRE, sh).startswith(
        "[shared:",
    )  # the whole request is the block
    short = "A heading\n\nbody"
    for i in range(3):
        shared_text.update_counts(ev, f"s{i}", short)
    assert (
        shared_text.mark(short, shared_text.shared(ev)) == short
    )  # under MIN_CHARS: stays inline


def test_a_block_in_under_half_the_episodes_is_not_shared(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for i in range(2):
        shared_text.update_counts(ev, f"a{i}", PRE + f"\n\nx{i}")
    for i in range(3):
        shared_text.update_counts(ev, f"b{i}", f"other request {i} " * 20)
    assert shared_text.shared(ev) == {}


def test_batch_map_lists_the_block_file_and_reads_show_the_marker(tmp_path):
    eps = {
        f"e{i}": _episode(f"e{i}", request=(PRE + f"\n\nTask {i}",)) for i in range(3)
    }
    turns = [
        _call("r", "read_episode", {"episode": "e1", "part": "request"}),
        _call("m", "read", {"path": "/inputs/batch_map.json"}),
    ]
    model = Turns(turns)
    _, _, writer = _writer(tmp_path, model, max_calls=10)
    writer.load = eps.__getitem__
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", list(eps), False), "p1"),
    )
    head, body = model.outputs["r"].split("\n", 1)
    assert head == "== request =="
    request = json.loads(body)
    assert request.startswith("[shared:") and request.endswith("Task 1")
    bmap = (
        json.loads(model.outputs["m"].split("\n[…")[0])
        if model.outputs["m"].startswith("{")
        else None
    )
    assert bmap is None or bmap["shared_blocks"][0]["file"].startswith(
        "/inputs/shared/",
    )
    assert out.coverage["parts_read"] == {"e1": ["request"]}  # the read log
    assert bm.part_text(eps["e1"], "request") != bm.part_text(
        eps["e1"],
        "request",
        writer._shared,
    )


def test_a_shared_block_is_listed_once_and_readable_as_its_own_part(
    tmp_path,
):
    """A shared block is listed once in shared_blocks and can be read as its own part; no row lists it."""
    eps = {
        f"e{i}": _episode(f"e{i}", request=(PRE + f"\n\nTask {i}",)) for i in range(3)
    }
    ev = EvidenceStore(tmp_path / "x.sqlite")
    for e, ep in eps.items():
        shared_text.update_counts(ev, e, ep.request[0])
    sh = shared_text.shared(ev)
    bid = next(b for b, t in sh.items() if t == PRE)
    assert shared_text.marker(bid) == f"[shared:{bid}]"
    bmap = bm.build_batch_map(eps.__getitem__, list(eps), sh)
    # listed once in shared_blocks, never as a part of any row
    assert [b["id"] for b in bmap["shared_blocks"]] == sorted(sh)
    assert all(
        "required_parts" not in row and row["request"].startswith(f"[shared:{bid}]")
        for row in bmap["episodes"]
    )
    assert json.loads(bm.part_text(eps["e0"], f"shared:{bid}", sh)) == PRE


def test_a_shared_line_inside_an_episode_specific_block_and_in_observations_is_marked(
    tmp_path,
):
    """At medium effort the writer re-read the line-level instructions that sit in blocks with task-specific text."""
    ev = EvidenceStore(tmp_path / "e.sqlite")
    line = "Reply with exactly one action JSON on the last line."
    obs = "Each demo request returns one more input-output pair of the task."
    for i in range(3):
        shared_text.update_counts(
            ev,
            f"e{i}",
            f"Task {i}: rotate the grid {i} times\n{line}",
            [obs, f"demo {i}"],
        )
    sh = shared_text.shared(ev)
    assert set(sh.values()) == {line, obs}
    lid = next(b for b, t in sh.items() if t == line)
    assert (
        shared_text.mark(f"Task 9: something\n{line}", sh)
        == f"Task 9: something\n[shared:{lid}]"
    )
    ep = _episode("e1", request=(f"Task 1: rotate the grid 1 times\n{line}", obs))
    assert "[shared:" in json.loads(bm.part_text(ep, "observation:0", sh))
