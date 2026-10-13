"""The pool index is a file the brief points to (never inlined in the first message), a deferral can add evidence
to an open entry the writer names, the WRITE brief says new items start on trial and are revised later, and
two read_episode refusals are gone.
"""

import json


from unify.memory import batch_map, pool, library_prompts as pv
from unify.memory.writer_pass import writer_tools
from tests.memory import library_example_world as w
from tests.memory.library_example_world import (
    library_example_world,
)  # noqa: F401 (a fixture)
from tests.memory.test_pool import EMPTY, INDEX, _defer, _finish_with, _open, _Stores
from unify.memory.episodes import Cell
from tests.memory.test_writer_pass import _call, _writer, needs_bwrap
from tests.memory.test_writer_library_tools import EPS, _episode, _pass

# --- (a) the open index in the first message ------------------------------------------------------------------


def test_the_first_message_points_to_the_pool_and_the_brief_stays_the_same(
    tmp_path,
    monkeypatch,
):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, _defer("e2"))
    _, m1 = st.run([], "p1")
    _finish_with(monkeypatch, EMPTY)
    _, m2 = st.run([_open("e2")], "p2")
    did = pool.entry_id("deferred", "e2")
    first = m2.seen[1]["content"]
    assert first.splitlines()[0].startswith("Pass p2: ")
    # no index, no entry line and no count is inlined; the brief names /inputs/pool/INDEX.md
    assert "Candidate pool" not in first and did not in first
    assert "/inputs/pool/INDEX.md" in m2.seen[0]["content"]
    assert (
        m1.seen[0] == m2.seen[0]
    )  # the system brief is the same bytes with 0 and 1 open entries


def test_a_long_pool_puts_no_list_in_the_first_message(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    p = st.pool()
    for i in range(120):
        p.upsert(pool.deferred(f"e{i:03d}", "a rule " * 20), "p0")
    p.save("p0")
    _finish_with(monkeypatch, EMPTY)
    _, model = st.run([_open("e2")], "p1")
    first = model.seen[1]["content"]
    assert (
        "Candidate pool" not in first
        and "120" not in first
        and "a rule a rule" not in first
    )
    assert "/inputs/pool/INDEX.md" in model.seen[0]["content"]


# --- (b) a revisit adds evidence to the entry the writer names ------------------------------------------------


def test_a_deferral_can_add_its_episode_to_an_open_entry(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, _defer("e2"))
    st.run([], "p1")
    did = pool.entry_id("deferred", "e2")
    _finish_with(
        monkeypatch,
        {
            **EMPTY,
            "pool": {
                "defer": [
                    {"id": did, "episodes": ["e1"], "reason": "the same fill again"},
                ],
            },
        },
    )
    out, model = st.run([], "p2", eids=("e1",))
    assert out.pool["extended"] == [did] and out.pool["added"] == 0
    e = st.pool().entries[did]
    assert e.status == "open" and e.provenance["episodes"] == ["e2", "e1"]
    assert e.data["evidence"] == [
        {"episode": "e1", "reason": "the same fill again", "pass": "p2"},
    ]
    assert (
        pool.entry_id("deferred", "e1") not in st.pool().entries
    )  # no second entry for the revisit
    _finish_with(monkeypatch, EMPTY)
    _, m3 = st.run([_call("r", "read", {"path": INDEX}), _open("e2")], "p3")
    assert (
        f"`{did}` deferred: a rule that may come back (2 episodes;" in m3.outputs["r"]
    )


def test_a_deferral_must_name_an_open_entry_or_new(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, _defer("e2"))
    st.run([], "p1")
    did = pool.entry_id("deferred", "e2")
    _finish_with(
        monkeypatch,
        {
            **EMPTY,
            "pool": {"defer": [{"id": "0" * 16, "episodes": ["e2"], "reason": "r"}]},
        },
    )
    out, model = st.run([_open("e2"), _call("f", "finish", {"summary": "s"})], "p2")
    assert (
        model.outputs["f"]
        == f"not finished: G1: pool.defer: {'0' * 16} is not an open entry"
    )
    assert out.pool["extended"] == [] and st.pool().entries[did].provenance[
        "episodes"
    ] == ["e2"]
    names = {t["function"]["name"] for t in writer_tools(library_mode=True, pool=True)}
    assert "pool" in names and "dismiss" not in names


# --- (c) the WRITE brief ---------------------------------------------------------------------------------------


def test_the_write_brief_says_new_items_are_on_trial_and_revised_later():
    text = pv.write_brief_now()
    sentence = (
        "A new item starts as `on-trial`; later passes and the curate pass revise, generalise or remove it as "
        "evidence arrives, so storing a rule that worked is not a final commitment."
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


def test_the_overview_shows_an_end_cell_that_is_not_a_required_part(
    tmp_path,
    monkeypatch,
):
    """A live pass's overview was refused with KeyError (<episode>, 'cell:25'): its end cell defines no function and
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
    from unify.memory.writer_tools import _wanted_parts

    assert _wanted_parts({"part": "request"}) == (
        [{"part": "request", "offset": 0}],
        None,
    )
    assert (
        json.dumps(_wanted_parts({"parts": "[request, 'cell:0']"})[0])
        == '["request", "cell:0"]'
    )
    assert _wanted_parts({}) == (None, None)  # the table of contents


# --- the brief's inputs, minimal note entries, shared modules and fixture slices ---------------------------------


from unify.memory import fixtures as fx  # noqa: E402
from unify.memory.manifest import ManifestError, parse_manifest  # noqa: E402


def _brief_of(tmp_path, **cfg):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, _, writer = _writer(tmp_path, None, library_mode=True, **cfg)
    return writer._brief_library_mode()


def test_the_write_brief_names_each_input_the_pass_prepares(tmp_path):
    full = _brief_of(tmp_path / "a", staging_root=str(tmp_path), fork=True, pool=True)
    inputs = full.split("# Inputs\n")[1].split("\n# Tools")[0]
    # proposals first, then the evidence behind them
    assert (
        inputs.index("/inputs/staging/")
        < inputs.index("/inputs/pool/")
        < inputs.index("/inputs/batch_map.json")
        < inputs.index("/inputs/episodes/")
    )
    assert "/inputs/s0" not in full  # the writer's path has no miner input
    plain = _brief_of(tmp_path / "b")
    # the default configuration caps the pass's spend, so finish's refusal cap is stated
    assert plain == pv.write_brief_now(finish_capped=True)
    assert "/inputs/staging" not in plain and "/inputs/pool" not in plain
    # a staging root without the fork stages nothing, so the brief names no staging
    fork_off = _brief_of(tmp_path / "c", staging_root=str(tmp_path), pool=True)
    assert "/inputs/staging" not in fork_off and "/inputs/pool/" in fork_off
    # byte-stable within a run: the same configuration gives the same bytes
    assert full == _brief_of(
        tmp_path / "d",
        staging_root=str(tmp_path),
        fork=True,
        pool=True,
    )


def test_the_optional_inputs_keep_the_brief_guards():
    text = pv.write_brief_now(staging=True, pool=True)
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
    for marker in pv.CURATE_ONLY:
        assert marker not in text.lower()
    inputs = text.split("# Inputs\n")[1].split("\n# Tools")[0].lower()
    assert " must " not in inputs and "task" not in inputs and "answer" not in inputs
    curate = pv.curate_brief_now()
    assert "/inputs/staging" not in curate and "/inputs/pool" not in curate


def test_the_end_condition_is_a_decision_not_a_count():
    for text in (pv.write_brief_now(), pv.write_brief_now(staging=True, pool=True)):
        assert "every episode in the batch" not in text and "dismiss" not in text
        assert (
            "you have built the best library items you can from this batch, with the proposals as material and the episodes as evidence"
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
                        "covers": [{"episode": "e1", "type": "cell", "cell": 0}],
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
    assert "keeps raw bytes start to end-1" in tool["function"]["description"]


HELPER = '''"""Shared account helpers."""


class MemoryInputError(ValueError):
    pass


def _need(response):
    if not isinstance(response, dict) or "user_id" not in response:
        raise MemoryInputError("an acct.me response with a user_id field is needed")
    return response
'''
USERS = f'''"""Accounts."""

from memory.acct.common import MemoryInputError, _need


def user_id(response: dict) -> str:
    """The account's user id, read from an acct.me response.

    Effect: read
    Example: {w.EXAMPLE_TEST}
    """
    return _need(response)["user_id"]
'''


def test_a_shared_module_lands_when_declared_as_support_and_is_refused_when_not(
    library_example_world,
):
    """In a live pass the writer's shared `memory/arc/errors.py` was refused as an undeclared change, so every module
    re-implemented its parser and MemoryInputError."""
    mem, _, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get, module=USERS)
    files = {**files, "memory/acct/common.py": HELPER}
    undeclared = gate().merge(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures),
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
        w.candidate(mem, files),
        w.manifest(fixtures, support=["memory/acct/common.py"]),
        "p2",
        "write",
        None,
        "0.01",
    )
    assert declared.passed, declared.reasons


# --- every input the texts point at exists ---------------------------------------------------------------------

_LIST_INPUTS = (
    "import os\n"
    "for root, dirs, files in os.walk('/inputs'):\n"
    "    print(root + '/')\n"
    "    for f in files:\n"
    "        print(os.path.join(root, f))\n"
)


def _named_inputs(text: str) -> set[str]:
    """The concrete /inputs paths *text* names: no `<placeholder>` path, and none whose sentence says it may be
    absent ("when present")."""
    import re

    out = set()
    for m in re.finditer(r"/inputs/[A-Za-z0-9_.\-/]*", text):
        path = m.group(0).rstrip(".,;:")
        if text[m.end() : m.end() + 1] == "<" or path.endswith("/<"):
            continue
        start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start()))
        end = min(
            x
            for x in (text.find(". ", m.end()), text.find("\n", m.end()), len(text))
            if x >= 0
        )
        if "when present" in text[start:end] or "if present" in text[start:end]:
            continue
        out.add(path)
    return out


@needs_bwrap
def test_every_input_path_the_brief_and_first_message_name_exists_in_the_pass(tmp_path):
    for name, cfg in (
        ("plain", {}),
        (
            "staged",
            {"staging_root": str(tmp_path / "staging"), "fork": True, "pool": True},
        ),
    ):
        turns = [
            _call("ls", "execute_code", {"code": _LIST_INPUTS}),
            _call("f", "finish", {"summary": "s"}),
        ]
        _, model = _pass(tmp_path / name, turns, **cfg)
        listed = set(model.outputs["ls"].split())
        named = _named_inputs(
            model.seen[0]["content"] + "\n" + model.seen[1]["content"],
        )
        assert named, name
        missing = sorted(
            p for p in named if p not in listed and p.rstrip("/") + "/" not in listed
        )
        assert missing == [], (name, missing)
