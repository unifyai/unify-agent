"""Cross-episode evidence of a function's examples: argument groundings from fixture views, the two-episode rule, the
literal lint and the rule check on grounded values, and the kept fixture provenance."""

import hashlib
import json
import sqlite3

from unify.memory import example_evidence as xe
from unify.memory import fixtures as fx
from unify.memory.library_prompts import EPISODES_FIX
from unify.memory.evidence import EvidenceStore
from unify.memory.example_calls import ArgRecord, CallKey, ExampleCall

ITEM = "memory.acct.users:user_id"
TEST = "memory/acct/tests/test_users.py::test_user_id"
A = "memory/acct/tests/data/me_a.json"  # from e1
A2 = "memory/acct/tests/data/me_a2.json"  # also from e1
B = "memory/acct/tests/data/me_b.json"  # from e2
VALUES = {A: {"user_id": "u-1"}, A2: {"user_id": "u-3"}, B: {"user_id": "u-2"}}
EPISODE = {A: "e1", A2: "e1", B: "e2"}


def _entry(path, **extra):
    return {
        "kind": "copy",
        "episode": EPISODE[path],
        "source": "0",
        "slice": None,
        "trust": 1,
        "sha256": "-",
    } | extra


VIEWS = {
    p: fx.value_views(p, json.dumps(v).encode(), _entry(p)) for p, v in VALUES.items()
}


def arg(name, value, *, sha=None, path=None):
    return ArgRecord(
        name,
        fx.value_digest(value) if sha is None else sha,
        type(value).__name__,
        value,
        False,
        path,
    )


def call(ordinal, *args, outcome="returned"):
    return ExampleCall(
        item=ITEM,
        key=CallKey(TEST, (), ordinal),
        node=TEST,
        file="memory/acct/tests/test_users.py",
        line=10 + ordinal,
        args=tuple(args),
        outcome=outcome,
        raised=None,
        result_sha=None,
        result_type=None,
        result_json=None,
        opened=tuple(VALUES),
    )


def check(calls, derived=None, typed=frozenset()):
    ground = xe.groundings(calls, VIEWS, derived or {})
    return xe.evidence_check(ITEM, calls, ground, set(typed))


def c1(n, ids):
    return [
        f"[examples:episodes] {ITEM}'s calls that return use recorded values from {n} episode(s) ({ids}); "
        + EPISODES_FIX,
    ]


# --- groundings -----------------------------------------------------------------------------------------------------


def test_an_argument_equal_to_a_fixture_view_is_copied_a_path_too_and_a_scalar_is_literal():
    got = xe.grounding(
        call(
            0,
            arg("response", VALUES[A]),
            arg("colour", 3),
            arg("where", A, path=A),
            arg("blob", None, sha=""),
        ),
        VIEWS,
        {},
    )
    assert got["response"] == xe.Grounding("copied", (A,), ("e1",))
    assert got["where"] == xe.Grounding("copied", (A,), ("e1",))
    assert got["colour"].kind == "literal" and got["blob"].kind == "literal"


def test_an_argument_the_perturbation_changed_is_derived_from_that_fixture():
    got = xe.grounding(call(0, arg("n", 7)), VIEWS, {"n": {B}})
    assert got["n"] == xe.Grounding("derived", (B,), ("e2",))
    unknown = ArgRecord("x", None, "Env", None, False, None)
    assert xe.grounding(call(1, unknown), VIEWS, {})["x"] == xe.LITERAL


def test_a_value_from_an_assembled_file_is_grounded_in_its_own_line_s_episode():
    rel = "memory/acct/tests/data/users.me.inputs.jsonl"
    data = "".join(
        json.dumps(
            {"input": v, "source": {"episode": e, "source": "0", "slice": None}},
            sort_keys=True,
        )
        + "\n"
        for v, e in (({"user_id": "u-8"}, "e5"), ({"user_id": "u-9"}, "e6"))
    ).encode()
    views = {rel: fx.value_views(rel, data, fx.assembled_entry(data))}
    got = xe.grounding(call(0, arg("response", {"user_id": "u-9"})), views, {})
    assert got["response"] == xe.Grounding("copied", (rel,), ("e6",))


# --- the episode rule: one episode is enough, more are recorded ---------------------------------------------------------------------------------------------


def test_examples_from_one_episode_pass_and_more_episodes_are_recorded():
    one = [call(0, arg("response", VALUES[A]))]
    assert check(one) == ([], {"episodes": ["e1"], "fixtures": [A]})
    two = one + [call(1, arg("response", VALUES[B]))]
    assert check(two) == ([], {"episodes": ["e1", "e2"], "fixtures": [A, B]})


def test_two_fixtures_from_one_episode_count_once():
    calls = [call(0, arg("response", VALUES[A])), call(1, arg("response", VALUES[A2]))]
    assert check(calls) == ([], {"episodes": ["e1"], "fixtures": [A, A2]})


def test_a_call_the_function_refused_or_raised_on_does_not_count():
    calls = [
        call(0, arg("response", VALUES[A])),
        call(1, arg("response", VALUES[B]), outcome="refused"),
        call(2, arg("response", VALUES[B]), outcome="raised"),
    ]
    assert check(calls) == ([], {"episodes": ["e1"], "fixtures": [A]})
    # with no call that returned, nothing counts and the function is refused
    assert check(calls[1:])[0] == c1(0, "none")


def test_derived_arguments_and_typed_episode_covers_count():
    calls = [call(0, arg("response", VALUES[A])), call(1, arg("n", 7))]
    assert check(calls, {CallKey(TEST, (), 1): {"n": {B}}})[0] == []
    assert check(calls[:1], typed={"e9"})[1]["episodes"] == ["e1", "e9"]
    assert check(calls[:1], typed={"e9"})[0] == []


def test_no_grounded_call_names_no_episode():
    assert check([call(0, arg("colour", 3))]) == (
        c1(0, "none"),
        {"episodes": [], "fixtures": []},
    )


def test_a_backfilled_entry_counts():
    path = "memory/acct/tests/data/old.json"
    data = json.dumps({"user_id": "u-5"}).encode()
    entry = {
        "kind": "copy",
        "episode": "e7",
        "source": "0",
        "slice": None,
        "trust": 1,
        "sha256": "-",
        "backfilled": True,
    }
    views = dict(VIEWS, **{path: fx.value_views(path, data, entry)})
    calls = [
        call(0, arg("response", VALUES[A])),
        call(1, arg("response", {"user_id": "u-5"})),
    ]
    ground = xe.groundings(calls, views, {})
    assert xe.evidence_check(ITEM, calls, ground, set())[1]["episodes"] == ["e1", "e7"]


# --- the literal lint and the rule check: texts and thresholds unchanged ---------------------------------------------

SOURCE = b'''def user_id(response):
    """Return the id."""
    if response["user_id"] == "u-1-special":
        return None
    return response["user_id"]
'''


def test_the_literal_lint_reads_grounded_arguments_of_accepted_calls():
    a = {"user_id": "u-1-special"}
    b = {"user_id": "u-2-other"}
    views = {
        A: fx.value_views(A, json.dumps(a).encode(), _entry(A)),
        B: fx.value_views(B, json.dumps(b).encode(), _entry(B)),
    }
    calls = [
        call(0, arg("response", a), arg("colour", 3)),
        call(1, arg("response", b)),
        call(2, arg("response", {"user_id": "u-0-refused"}), outcome="refused"),
    ]
    ground = xe.groundings(calls, views, {})
    assert xe.lint_values(calls, ground) == [{"response": a}, {"response": b}]
    reasons, notes = xe.lint(ITEM, SOURCE, calls, ground)
    assert reasons == [
        f"{ITEM} line 3: a string of 11 characters literal equals a value that varies across its recorded inputs; "
        "make it a parameter",
    ]
    assert notes == []
    assert xe.lint(ITEM, SOURCE, calls[:1], ground) == (
        [],
        [
            f"G2 literal lint: {ITEM} has fewer than 2 recorded inputs, so nothing varies yet",
        ],
    )
    # a typed episode cover's parameters are recorded inputs too
    assert (
        xe.lint(ITEM, SOURCE, calls[:1], ground, extra=[{"response": b}])[0] == reasons
    )


def test_the_rule_check_needs_examples_from_one_episode_in_the_library_layout():
    from unify.memory.gate import RULE_EPISODES

    # the earlier layout keeps its own count for covers; the library layout asks one episode, as of every function
    assert RULE_EPISODES == 2 and xe.LIBRARY_RULE_EPISODES == 1
    path = "memory/acct/users.py"
    want = (
        f"{ITEM} replaces a value it computed from its input under a condition (line 3 of {path}); a function "
        "that encodes such a rule needs examples from at least 1 episode"
    )
    assert xe.rule_problems(
        ITEM,
        path=path,
        rule_line=3,
        unchecked=False,
        episodes=set(),
    ) == ([want], [])
    assert xe.rule_problems(
        ITEM,
        path=path,
        rule_line=3,
        unchecked=False,
        episodes={"e1"},
    ) == ([], [])
    assert xe.rule_problems(
        ITEM,
        path=path,
        rule_line=3,
        unchecked=False,
        episodes={"e1", "e2"},
    ) == ([], [])
    assert xe.rule_problems(
        ITEM,
        path=path,
        rule_line=None,
        unchecked=False,
        episodes=set(),
    ) == ([], [])
    assert xe.rule_problems(
        ITEM,
        path=path,
        rule_line=3,
        unchecked=True,
        episodes=set(),
    ) == (
        [],
        [f"G2 rule check: {ITEM} is too large to analyse; not checked"],
    )
    calls = [
        call(0, arg("response", VALUES[A])),
        call(1, arg("response", VALUES[B]), outcome="refused"),
    ]
    ground = xe.groundings(calls, VIEWS, {})
    assert xe.rule_episodes(ground) == {"e1", "e2"}
    assert xe.rule_episodes(ground, calls) == {"e1"}


# --- the kept fixture provenance --------------------------------------------------------------------------------------


def _tables(path):
    con = sqlite3.connect(str(path))
    try:
        return {
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        con.close()


def test_fixture_provenance_is_created_lazily_and_keyed_by_path_and_bytes(tmp_path):
    db = tmp_path / "e.sqlite"
    ev = EvidenceStore(db)
    assert ev.fixture_provenance(A, "aa") is None
    assert "fixture_provenance" not in _tables(db)
    first = _entry(A, sha256="aa")
    ev.add_fixture_provenance(A, "aa", first, "c1")
    assert "fixture_provenance" in _tables(db)
    assert ev.fixture_provenance(A, "aa") == first
    # the first landed entry for the same bytes is kept
    ev.add_fixture_provenance(A, "aa", _entry(B, sha256="aa"), "c2")
    assert ev.fixture_provenance(A, "aa") == first
    assert (
        ev.fixture_provenance(A, "bb") is None
        and ev.fixture_provenance(B, "aa") is None
    )
    con = sqlite3.connect(str(db))
    try:
        stored = con.execute(
            "SELECT entry, commit_sha FROM fixture_provenance",
        ).fetchall()
    finally:
        con.close()
    assert stored == [(json.dumps(first, sort_keys=True, separators=(",", ":")), "c1")]


def test_a_kept_row_is_provenance_only_for_the_same_bytes(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    data = json.dumps(VALUES[A]).encode()
    sha = hashlib.sha256(data).hexdigest()
    ev.add_fixture_provenance(A, sha, _entry(A, sha256=sha), "c1")
    ledger, notes = fx.ledger_for(
        {A: data},
        {},
        ev.fixture_provenance,
        lambda e: None,
        lambda s: b"",
        [],
    )
    assert ledger == {A: _entry(A, sha256=sha)} and notes == []
    edited = data + b" "
    ledger, notes = fx.ledger_for(
        {A: edited},
        {},
        ev.fixture_provenance,
        lambda e: None,
        lambda s: b"",
        [],
    )
    assert ledger == {} and "has no recorded provenance" in notes[0]
