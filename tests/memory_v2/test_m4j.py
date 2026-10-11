"""Memory v2.1 m4j (MAIN-54 for the lead, 11 Oct; diagnosis a1-m4i-s25-pool-diagnosis-20261011.md): the open pool
index is shown in the first message, a deferral can add evidence to an open entry the writer names, the WRITE brief
says new items start as experimental and are revised later, and two read_episode refusals are gone.
"""

import json

from unify.memory_v2 import batch_map, pool, prompts_v21 as pv, sol_pass
from unify.memory_v2.sol_pass import sol_tools
from unify.memory_v2.views import Coverage
from tests.memory_v2.test_pool import EMPTY, INDEX, _defer, _finish_with, _Stores
from unify.memory_v2.episodes import Cell
from tests.memory_v2.test_sol_pass import _call
from tests.memory_v2.test_sol_v21_tools import EPS, _episode, _pass

# --- (a) the open index in the first message ------------------------------------------------------------------


def test_the_first_message_shows_the_open_index_and_the_brief_stays_the_same(
    tmp_path,
    monkeypatch,
):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    _, m1 = st.run([_defer("e2")], "p1")
    _, m2 = st.run([_call("x", "dismiss", {"episode": "e2", "reason": "seen"})], "p2")
    did = pool.entry_id("deferred", "e2")
    first = m2.seen[1]["content"]
    assert first.splitlines()[0].startswith("Pass p2: ")
    shown = first.split("read it at a marker's offset for more):\n\n", 1)[1]
    assert shown.startswith("# Candidate pool: open entries\n")
    assert (
        f"- `{did}` deferred: episode e2 deferred: a rule that may come back (1 episode(s): e2; opened in p1)"
        in shown
    )
    assert (
        m1.seen[0] == m2.seen[0]
    )  # the system brief is the same bytes with 0 and 1 open entries


def test_a_long_index_is_shown_one_page_with_its_next_offset(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    p = st.pool()
    for i in range(120):
        p.upsert(pool.deferred(f"e{i:03d}", "a rule " * 20), "p0")
    p.save("p0")
    _finish_with(monkeypatch, EMPTY)
    _, model = st.run(
        [_call("x", "dismiss", {"episode": "e2", "reason": "seen"})],
        "p1",
    )
    shown = model.seen[1]["content"].split(
        "read it at a marker's offset for more):\n\n",
        1,
    )[1]
    assert "[… shown bytes 0–" in shown and "; next: offset=" in shown
    assert len(shown.encode()) < 8300


# --- (b) a revisit adds evidence to the entry the writer names ------------------------------------------------


def test_a_deferral_can_add_its_episode_to_an_open_entry(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    st.run([_defer("e2")], "p1")
    did = pool.entry_id("deferred", "e2")
    turn = _call(
        "d",
        "dismiss",
        {
            "episode": "e1",
            "reason": "the same fill again",
            "defer": True,
            "pool_entry": did,
        },
    )
    out, model = st.run([turn], "p2", eids=("e1",))
    assert (
        model.outputs["d"] == "ok"
        and out.pool["extended"] == [did]
        and out.pool["added"] == 0
    )
    e = st.pool().entries[did]
    assert e.status == "open" and e.provenance["episodes"] == ["e2", "e1"]
    assert e.data["evidence"] == [
        {"episode": "e1", "reason": "the same fill again", "pass": "p2"},
    ]
    assert (
        pool.entry_id("deferred", "e1") not in st.pool().entries
    )  # no second entry for the revisit
    _, m3 = st.run(
        [
            _call("r", "read", {"path": INDEX}),
            _call("x", "dismiss", {"episode": "e2", "reason": "s"}),
        ],
        "p3",
    )
    assert (
        f"`{did}` deferred: episode e2 deferred: a rule that may come back (2 episode(s): e2, e1;"
        in m3.outputs["r"]
    )


def test_pool_entry_must_name_an_open_entry_and_needs_defer(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, EMPTY)
    st.run([_defer("e2")], "p1")
    did = pool.entry_id("deferred", "e2")
    turns = [
        _call(
            "u",
            "dismiss",
            {"episode": "e2", "reason": "r", "defer": True, "pool_entry": "0" * 16},
        ),
        _call("n", "dismiss", {"episode": "e2", "reason": "r", "pool_entry": did}),
        _call("ok", "dismiss", {"episode": "e2", "reason": "seen"}),
    ]
    out, model = st.run(turns, "p2")
    assert (
        model.outputs["u"]
        == f"refused: {'0' * 16!r} is not an open pool entry; nothing was dismissed"
    )
    assert (
        model.outputs["n"]
        == "refused: pool_entry needs defer=true; nothing was dismissed"
    )
    assert out.pool["extended"] == [] and st.pool().entries[did].provenance[
        "episodes"
    ] == ["e2"]
    dismiss = next(
        t for t in sol_tools(v21=True, pool=True) if t["function"]["name"] == "dismiss"
    )
    assert set(dismiss["function"]["parameters"]["properties"]) == {
        "episode",
        "reason",
        "defer",
        "pool_entry",
    }
    assert dismiss["function"]["parameters"]["required"] == ["episode", "reason"]
    plain = next(t for t in sol_tools(v21=True) if t["function"]["name"] == "dismiss")
    assert set(plain["function"]["parameters"]["properties"]) == {"episode", "reason"}


# --- (c) the WRITE brief ---------------------------------------------------------------------------------------


def test_the_write_brief_says_new_items_are_experimental_and_revised_later():
    text = pv.write_brief_now()
    sentence = (
        "A new item starts as `experimental`; later passes and CURATE revise, generalise or remove it as evidence "
        "arrives, so storing a rule that worked is not a final commitment."
    )
    assert sentence in text and sentence not in pv.curate_brief_now()
    line = text.split("A new item starts as")[1].split("\n")[0]
    assert (
        "must" not in line
        and "only" not in line
        and "task" not in line
        and " id" not in line
    )
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []


# --- (d) read_episode ------------------------------------------------------------------------------------------


def test_a_part_that_is_not_required_is_never_done_and_never_raises():
    cov = Coverage({"e1": ["request"]}, {("e1", "request"): 3})
    assert cov.part_done("e1", "cell:25") is False
    cov.credit("e1", "cell:25", 0, 10)
    assert cov.part_done("e1", "cell:25") is False and cov.missing() == ["e1"]


def test_the_overview_shows_an_end_cell_that_is_not_a_required_part(
    tmp_path,
    monkeypatch,
):
    """Live m4i s25: overview refused with KeyError (<episode>, 'cell:25'): its end cell defines no function and
    carries no signal, so it is not a required part, and crediting it raised."""
    monkeypatch.setitem(EPS, "e7", _episode("e7", [Cell(0, "x = 1\nprint(x)", "1")]))
    assert "cell:0" in batch_map.solution_first(EPS["e7"])[1]
    _, model = _pass(
        tmp_path,
        [_call("o", "read_episode", {"episode": "e7", "part": "overview"})],
        eids=("e7",),
        max_calls=10,
    )
    assert (
        not model.outputs["o"].startswith("refused") and "x = 1" in model.outputs["o"]
    )


def test_parts_given_as_text_are_read_as_the_list_they_name(tmp_path):
    forms = {
        "list": {"parts": ["request"]},
        "json": {"parts": '["request"]'},
        "prefixed": {"part": 'parts=["request"]'},
        "commas": {"part": "request,request"},
        "one": {"parts": "request"},
    }
    turns = [_call(k, "read_episode", {"episode": "e1", **v}) for k, v in forms.items()]
    _, model = _pass(tmp_path, turns, max_calls=12)
    for k in forms:
        assert model.outputs[k].startswith("== request =="), (k, model.outputs[k][:120])
    assert sol_pass._parts_from_text({"part": "shared:0affb0d119ea"}) == {
        "part": "shared:0affb0d119ea",
    }
    assert sol_pass._parts_from_text({"part": "request"}) == {"part": "request"}
    assert (
        json.dumps(sol_pass._parts_from_text({"parts": "[request, 'cell:0']"})["parts"])
        == '["request", "cell:0"]'
    )
