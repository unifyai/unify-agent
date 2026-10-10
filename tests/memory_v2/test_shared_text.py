"""Memory v2.1 r5 (r4 §1): request text shared by most episodes is shown once, as a marker with its file."""

import asyncio
import json

from unify.memory_v2 import batch_map as bm
from unify.memory_v2 import shared_text
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import Turns, _call, _sol
from tests.memory_v2.test_sol_v21_tools import _episode

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
    assert list(sh.values()) == [PRE]
    marked = shared_text.mark(PRE + "\n\nTask 1: do thing 1", sh)
    bid = next(iter(sh))
    assert marked == shared_text.marker(bid) + "\n\nTask 1: do thing 1"


def test_shared_blocks_edge_cases(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert shared_text.shared(ev) == {}  # nothing seen: no table, nothing shared
    shared_text.update_counts(ev, "e0", PRE)
    assert shared_text.shared(ev) == {}  # one episode: nothing is shared
    shared_text.update_counts(ev, "e1", PRE)
    sh = shared_text.shared(ev)
    assert shared_text.mark(PRE, sh).startswith(
        "[shared request text ",
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
    _, _, sol = _sol(tmp_path, model, v21=True, max_calls=10)
    sol.load = eps.__getitem__
    out = asyncio.run(
        sol.run(PassRequest("incremental", "svc", list(eps), False), "p1"),
    )
    request = json.loads(model.outputs["r"])
    assert request.startswith("[shared request text ") and request.endswith("Task 1")
    bmap = (
        json.loads(model.outputs["m"].split("\n[…")[0])
        if model.outputs["m"].startswith("{")
        else None
    )
    assert bmap is None or bmap["shared_blocks"][0]["file"].startswith(
        "/inputs/shared/",
    )
    assert (
        out.coverage["parts_read"] >= 1
    )  # the marked request was read in full: covered
    assert bm.part_text(eps["e1"], "request") != bm.part_text(
        eps["e1"],
        "request",
        sol._shared,
    )


def test_a_shared_block_is_a_required_part_shown_once_then_credited_as_identical(
    tmp_path,
):
    """RUNTIME B2: the marker alone never covers the block; it is a required part, read once, then credited
    elsewhere by the identical-part credit."""
    eps = {
        f"e{i}": _episode(f"e{i}", request=(PRE + f"\n\nTask {i}",)) for i in range(3)
    }
    ev = EvidenceStore(tmp_path / "x.sqlite")
    for e, ep in eps.items():
        shared_text.update_counts(ev, e, ep.request[0])
    sh = shared_text.shared(ev)
    bid = next(iter(sh))
    assert f"'shared:{bid}'" in shared_text.marker(bid)
    bmap = bm.build_batch_map(eps.__getitem__, list(eps), sh)
    assert all(f"shared:{bid}" in row["required_parts"] for row in bmap["episodes"])
    assert json.loads(bm.part_text(eps["e0"], f"shared:{bid}", sh)) == PRE
