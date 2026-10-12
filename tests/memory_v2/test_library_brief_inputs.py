"""Memory v2.1 m4j (MAIN-54 for the lead, 11 Oct; diagnosis a1-m4i-s25-pool-diagnosis-20261011.md): the open pool
index is shown in the first message, a deferral can add evidence to an open entry the writer names, the WRITE brief
says new items start as experimental and are revised later, and two read_episode refusals are gone.
"""

import json

from unify.memory_v2 import batch_map, pool, library_prompts as pv, writer_pass
from unify.memory_v2.writer_pass import writer_tools
from unify.memory_v2.views import Coverage
from tests.memory_v2.test_gate import _candidate
from tests.memory_v2.test_gate_library_mode import (
    _files_and_fixtures,
    _man,
    world21,
)  # noqa: F401 (a fixture)
from tests.memory_v2.test_pool import EMPTY, INDEX, _defer, _finish_with, _Stores
from unify.memory_v2.episodes import Cell
from tests.memory_v2.test_writer_pass import _call, _writer
from tests.memory_v2.test_writer_library_tools import EPS, _episode, _pass

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
        f"- `{did}` deferred: a rule that may come back (1 episode; opened in p1)"
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
        f"`{did}` deferred: a rule that may come back (2 episodes;" in m3.outputs["r"]
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
        t
        for t in writer_tools(library_mode=True, pool=True)
        if t["function"]["name"] == "dismiss"
    )
    assert set(dismiss["function"]["parameters"]["properties"]) == {
        "episode",
        "reason",
        "defer",
        "pool_entry",
    }
    assert dismiss["function"]["parameters"]["required"] == ["episode", "reason"]
    plain = next(
        t for t in writer_tools(library_mode=True) if t["function"]["name"] == "dismiss"
    )
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
    assert writer_pass._parts_from_text({"part": "shared:0affb0d119ea"}) == {
        "part": "shared:0affb0d119ea",
    }
    assert writer_pass._parts_from_text({"part": "request"}) == {"part": "request"}
    assert (
        json.dumps(
            writer_pass._parts_from_text({"parts": "[request, 'cell:0']"})["parts"],
        )
        == '["request", "cell:0"]'
    )


# --- m4k folded into m4j (MAIN for the lead, 11 Oct; writer audit writer-audit-m4i-20261011T1238Z) ---------------


from unify.memory_v2 import fixtures as fx  # noqa: E402
from unify.memory_v2.manifest import ManifestError, parse_manifest  # noqa: E402

ALL_INPUTS = (pv.STAGING_INPUT, pv.POOL_INPUT)


def _brief_of(tmp_path, **cfg):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, _, writer = _writer(tmp_path, None, library_mode=True, **cfg)
    return writer._brief_library_mode()


def test_the_write_brief_names_each_input_the_pass_prepares(tmp_path):
    full = _brief_of(
        tmp_path / "a",
        staging_root=str(tmp_path),
        pool=True,
    )
    for bullet in ALL_INPUTS:
        assert bullet in full
    inputs = full.split("# Inputs\n")[1].split("\n# Tools")[0]
    assert (
        inputs.index("/inputs/gate/previous.md")
        < inputs.index("/inputs/staging/")
        < inputs.index("/inputs/pool/")
    )
    assert "/inputs/s0" not in full  # the writer's path has no miner input
    plain = _brief_of(tmp_path / "b")
    assert plain == pv.write_brief_now() and not any(b in plain for b in ALL_INPUTS)
    fork_off = _brief_of(
        tmp_path / "c",
        staging_root=str(tmp_path),
    )
    assert pv.POOL_INPUT not in fork_off and pv.STAGING_INPUT in fork_off
    # byte-stable within a run: the same configuration gives the same bytes
    assert full == _brief_of(
        tmp_path / "d",
        staging_root=str(tmp_path),
        pool=True,
    )


def test_the_new_input_bullets_keep_the_brief_guards():
    text = pv.write_brief_now(extra_inputs=ALL_INPUTS, pool=True)
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
    for marker in pv.CURATE_ONLY:
        assert marker not in text.lower()
    for bullet in ALL_INPUTS:
        low = bullet.lower()
        assert (
            " must " not in low
            and " only " not in low
            and "task" not in low
            and "answer" not in low
        )
    assert all(b not in pv.curate_brief_now() for b in ALL_INPUTS)


def test_the_end_condition_does_not_make_dismissing_everything_the_easy_path():
    off, on = pv.write_brief_now(), pv.write_brief_now(pool=True)
    assert (
        "every episode in the batch is covered or dismissed with a reason, and every item"
        in off
    )
    assert (
        "every episode in the batch is covered, dismissed with a reason, or attached to a pool entry"
        in on
    )
    for text in (off, on):
        assert "dismiss the rest" not in text and "holding nothing to store" not in text
        assert (
            "read in full what may be worth keeping, and decide each episode on what it holds"
            in text
        )
        assert "kept as a draft" in text  # the drafts rule's phrase stays (RULES)


def test_a_note_takes_a_minimal_manifest_entry_and_the_brief_shows_it():
    text = pv.write_brief_now()
    assert (
        '{"item": "notes/<topic>/<slug>.md", "kind": "note", "source_episodes": [...]}'
        in text
    )
    man = parse_manifest(
        {
            "items": [
                {
                    "item": "notes/arc/fill.md",
                    "kind": "note",
                    "source_episodes": ["e1"],
                },
            ],
        },
        library_mode=True,
    )
    assert [(i.item, i.kind) for i in man.items] == [("notes/arc/fill.md", "note")]
    try:
        parse_manifest(
            {
                "items": [
                    {
                        "item": "notes/arc/fill.md",
                        "kind": "note",
                        "covers": [["e1", 0]],
                    },
                ],
            },
            library_mode=True,
        )
    except ManifestError as exc:
        assert "only functions cover recorded work" in str(exc)
    else:
        raise AssertionError("a note with covers was accepted")


def test_support_may_declare_a_shared_module_of_the_package():
    man = parse_manifest(
        {
            "items": [],
            "support": ["memory/arc/common.py", "memory/arc/tests/helpers.py"],
        },
        library_mode=True,
    )
    assert man.support == ["memory/arc/common.py", "memory/arc/tests/helpers.py"]
    for bad in ("notes/arc/x.md", "memory/common.py", "memory/arc/tests/data/x.json"):
        try:
            parse_manifest({"items": [], "support": [bad]}, library_mode=True)
        except ManifestError as exc:
            assert "is not a test helper" in str(exc)
        else:
            raise AssertionError(f"{bad} was accepted as support")
    assert (
        "a shared module `memory/<package>/<module>.py` that the package's items import"
        in pv.write_brief_now()
    )


def test_a_fixture_slice_refusal_names_the_valid_range():
    try:
        fx._cut(b"x" * 1950, [0, 3831])
    except fx.FixtureError as exc:
        msg = str(exc)
    else:
        raise AssertionError("an out-of-range slice was accepted")
    assert msg.startswith("slice [0, 3831] is outside the 1950 recorded bytes")
    assert "0 <= start < end <= 1950; [0, 1950] or no slice copies all of it" in msg
    assert fx._cut(b"abc", [0, 3]) == b"abc"
    tool = next(
        t for t in writer_tools(library_mode=True) if t["function"]["name"] == "fixture"
    )
    assert (
        "keeps bytes start to end-1 of the recorded value's UTF-8 bytes"
        in tool["function"]["description"]
    )
    assert "byte offsets into the recorded value's UTF-8 bytes" in pv.write_brief_now()


HELPER = '''"""Shared account helpers."""


class MemoryInputError(ValueError):
    pass


def _need(observation):
    if not isinstance(observation, dict) or "user_id" not in observation:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return observation
'''
USERS = '''"""Accounts."""

from memory.acct.common import MemoryInputError, _need


def user_id(observation):
    """The account's user id.

    Effect: read
    Input: observation
    """
    return _need(observation)["user_id"]
'''


def test_a_shared_module_lands_when_declared_as_support_and_is_refused_when_not(
    world21,
):
    """Live m4i: the writer's shared `memory/arc/errors.py` was refused as an undeclared change, so every module
    re-implemented its parser and MemoryInputError."""
    mem, _, blobs, gate = world21
    files, fixtures = _files_and_fixtures(blobs)
    files = {**files, "memory/acct/users.py": USERS, "memory/acct/common.py": HELPER}
    undeclared = gate().merge(
        mem.head(),
        _candidate(mem, files),
        _man(fixtures),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert not undeclared.passed and any(
        "undeclared change memory/acct/common.py" in r for r in undeclared.reasons
    )
    declared = gate().merge(
        mem.head(),
        _candidate(mem, files),
        {**_man(fixtures), "support": ["memory/acct/common.py"]},
        "p2",
        "write",
        None,
        "0.01",
    )
    assert declared.passed, declared.reasons
