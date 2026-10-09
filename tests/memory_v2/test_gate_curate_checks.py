"""CURATE's gate rules on stub runs (P6; spec v2.1 §10.4, §9.1): no jail, no git."""

from types import SimpleNamespace

import pytest

from unify.memory_v2.gate_v21 import V21Checks, V21Config
from tests.memory_v2.test_gate_v21_checks import _item, _Run
from tests.memory_v2.test_layout import LIB, _tree
from tests.memory_v2.test_overlap import SPLIT, SPLIT_ID, SPLIT_PATH, TOKENS

PARENT = "p" * 40
T_SPLIT = "memory/text/tests/test_split.py"
SPLIT_TEST = "from memory.text.split import split_words\n\n\ndef test_split_words():\n    assert split_words('a b') == ['a', 'b']\n"
SPLIT_ALIAS = '"""Splitting: split_words is kept as an alias."""\n\nfrom memory.text.parse import tokens as split_words\n'
BEFORE = {**LIB, SPLIT_PATH: SPLIT, T_SPLIT: SPLIT_TEST}
AFTER = {**LIB, SPLIT_PATH: SPLIT_ALIAS, T_SPLIT: SPLIT_TEST}
WHY = "split_words repeated tokens"
OK = {"why": WHY, "aliases": {SPLIT_ID: TOKENS}}


def _checks(
    tmp_path,
    parent,
    cand,
    raw,
    *,
    items=(),
    deleted=(),
    deleted_tests=(),
    role="curate",
    live=None,
):
    p_tree, c_tree = _tree(tmp_path / "p", parent), _tree(tmp_path / "c", cand)
    gate = SimpleNamespace(
        v21=V21Config(role=role),
        ev=SimpleNamespace(aliases=lambda: dict(live or {})),
    )
    man = SimpleNamespace(
        items=list(items),
        deleted=list(deleted),
        deleted_tests=list(deleted_tests),
        unlisted=[],
        skeleton=[],
    )
    run = _Run(
        man=man,
        fails=[],
        notes=[],
        verification={},
        curate_due=False,
        curations=[],
        tmp=tmp_path,
        changed=sorted(
            p for p in set(parent) | set(cand) if parent.get(p) != cand.get(p)
        ),
        p_files={k: ("100644", "x") for k in parent},
        c_files={k: ("100644", "x") for k in cand},
        p_tree=p_tree,
        c_tree=c_tree,
        p_bodies={},
        c_bodies={},
        manifest_raw=raw,
        parent=PARENT,
    )
    return V21Checks(gate, run)


def _merge(tmp_path, raw=OK, **kw):
    kw.setdefault("items", [_item(item=TOKENS, tests=[T_SPLIT])])
    kw.setdefault("deleted", [SPLIT_ID])
    v = _checks(tmp_path, BEFORE, AFTER, raw, **kw)
    v.curate()
    return v


def _reasons(v):
    return [r for _, r, _ in v.run.fails]


def test_a_clean_alias_merge_has_no_failure_and_records_its_curation(tmp_path):
    v = _merge(tmp_path)
    assert v.run.fails == []
    assert v.run.curations == [
        {"item": SPLIT_ID, "action": "alias", "target": TOKENS, "reason": WHY},
    ]


def test_write_may_not_declare_aliases_or_retirements(tmp_path):
    v = _merge(tmp_path, role="write")
    assert v.run.fails == [
        (
            "G1",
            "aliases and retirements belong to CURATE; a WRITE pass may not declare them",
            None,
        ),
    ]
    assert v.run.curations == []


@pytest.mark.parametrize("why", [None, " ", "two\nlines"])
def test_curate_needs_a_one_line_why(tmp_path, why):
    raw = {"aliases": {SPLIT_ID: TOKENS}, **({} if why is None else {"why": why})}
    assert _reasons(_merge(tmp_path, raw=raw)) == [
        "a CURATE pass states why in one line (the manifest's why)",
    ]


@pytest.mark.parametrize(
    "aliases, cand, reason",
    [
        (
            {"split_words": TOKENS},
            AFTER,
            f"alias split_words -> {TOKENS}: both must be function ids (memory.<package>.<module>:<name>)",
        ),
        (
            {SPLIT_ID: "memory.text.parse:gone"},
            AFTER,
            f"alias {SPLIT_ID} -> memory.text.parse:gone: the candidate holds no function memory.text.parse:gone",
        ),
        (
            {SPLIT_ID: TOKENS, TOKENS: "memory.text.parse:words"},
            AFTER,
            f"alias {SPLIT_ID} -> {TOKENS}: an alias forwards to an item that is not itself an alias",
        ),
        (
            {SPLIT_ID: TOKENS},
            BEFORE,
            f"alias {SPLIT_ID}: its module does not forward split_words to {TOKENS} (bind it to the function, "
            "import it under that name, or make it a one-line wrapper that returns tokens(...))",
        ),
        (
            {"memory.text.split:brand_new": TOKENS},
            {**AFTER, SPLIT_PATH: SPLIT_ALIAS + "brand_new = split_words\n"},
            "alias memory.text.split:brand_new: the parent has no item memory.text.split:brand_new to keep working",
        ),
    ],
)
def test_an_alias_forwards_an_old_name_to_an_existing_function(
    tmp_path,
    aliases,
    cand,
    reason,
):
    v = _checks(
        tmp_path,
        BEFORE,
        cand,
        {"why": WHY, "aliases": aliases},
        items=[_item(item=TOKENS, tests=[T_SPLIT])],
        deleted=[SPLIT_ID],
    )
    v.curate()
    assert reason in _reasons(v)


def test_every_deleted_item_is_aliased_or_retired_with_a_reason(tmp_path):
    cand = dict(LIB)  # split.py and its test are gone
    for n, retired in enumerate(({}, {SPLIT_ID: " "})):
        v = _checks(
            tmp_path / str(n),
            BEFORE,
            cand,
            {"why": WHY, "retired": retired},
            deleted=[SPLIT_ID],
            deleted_tests=[T_SPLIT],
        )
        v.curate()
        assert _reasons(v) == [
            f"deleted item {SPLIT_ID} needs a reason: keep its name as an alias (aliases) or retire it with a "
            "one-line reason (retired)",
        ]
    raw = {
        "why": WHY,
        "retired": {
            SPLIT_ID: "tokens does the same job",
            "memory.text.parse:words": "unused",
        },
    }
    v = _checks(
        tmp_path / "ok",
        BEFORE,
        cand,
        raw,
        deleted=[SPLIT_ID],
        deleted_tests=[T_SPLIT],
    )
    v.curate()
    assert _reasons(v) == ["retired memory.text.parse:words is not in deleted"]
    assert v.run.curations == [
        {
            "item": SPLIT_ID,
            "action": "retire",
            "target": None,
            "reason": "tokens does the same job",
        },
    ]


def test_the_old_names_tests_are_listed_under_the_target(tmp_path):
    want = f"the tests of {SPLIT_ID} ({T_SPLIT}) keep testing it through its alias: list them under {TOKENS} in items"
    v = _merge(tmp_path / "a", items=[_item(item=TOKENS, tests=[])])
    assert v.run.fails == [("G3", want, TOKENS)]
    v = _merge(tmp_path / "b", items=[])
    assert v.run.fails == [("G3", want, None)]


def test_an_alias_stays_for_one_more_pass(tmp_path):
    before, after = {**LIB, SPLIT_PATH: SPLIT_ALIAS}, dict(LIB)
    raw = {"why": "drop the old name"}
    young = {SPLIT_ID: {"target": TOKENS, "pass_id": "c0", "commit": PARENT}}
    v = _checks(tmp_path / "young", before, after, raw, live=young)
    v.curate()
    assert v.run.fails == [
        (
            "G5",
            f"alias {SPLIT_ID} was added by the pass that made the parent ({PARENT[:12]}); it stays for one "
            "more pass",
            None,
        ),
    ]
    old = {SPLIT_ID: {**young[SPLIT_ID], "commit": "o" * 40}}
    v = _checks(tmp_path / "old", before, after, raw, live=old)
    v.curate()
    assert v.run.fails == []
    assert v.run.curations == [
        {
            "item": SPLIT_ID,
            "action": "drop_alias",
            "target": TOKENS,
            "reason": "drop the old name",
        },
    ]


def test_curate_may_not_grow_the_library(tmp_path):
    extra = {
        **BEFORE,
        "memory/text/extra.py": '"""Extra."""\n\n\ndef extra(s):\n    """Echo."""\n    return s\n',
    }
    v = _checks(
        tmp_path,
        BEFORE,
        extra,
        {"why": "a new reader"},
        items=[_item(item="memory.text.extra:extra")],
    )
    v.curate()
    assert v.run.fails == [
        (
            "G5",
            "a CURATE pass may not grow the library (8 to 9 items that are not aliases); add a function only "
            "as the target that old names become aliases of",
            None,
        ),
    ]


def test_a_declared_wrapper_alias_is_not_an_edited_function(tmp_path):
    wrapper = (
        '"""Splitting."""\nfrom memory.text.parse import tokens\n\n\n'
        'def split_words(text):\n    """Kept as an alias."""\n    return tokens(text)\n'
    )
    for role, edited in (("curate", []), ("write", [SPLIT_ID])):
        v = _checks(
            tmp_path / role,
            BEFORE,
            {**BEFORE, SPLIT_PATH: wrapper},
            {"why": WHY, "aliases": {SPLIT_ID: TOKENS}},
            items=[
                _item(item=SPLIT_ID, tests=[T_SPLIT]),
                _item(item=TOKENS, tests=[T_SPLIT]),
            ],
            role=role,
        )
        v.run.p_bodies = {SPLIT_ID: ("old", "def", True)}
        v.run.c_bodies = {SPLIT_ID: ("new", "def", True)}
        assert [it.item for it in v._edited()] == edited


def test_the_static_checks_include_curates(tmp_path, monkeypatch):
    seen = []
    for name in ("g1", "_plain", "_test_changes"):
        monkeypatch.setattr(V21Checks, name, lambda self: None)
    monkeypatch.setattr(V21Checks, "curate", lambda self: seen.append("curate"))
    _checks(tmp_path, BEFORE, AFTER, OK).static()
    assert seen == ["curate"]


# -- P6 Amendment B: G5 keeps procedures' typed covers through a remaining item's re-run (D28) ------------------
import json  # noqa: E402

from unify.memory_v2 import gate_v21 as gv  # noqa: E402
from unify.memory_v2.procedures import Outcome  # noqa: E402

COVER = json.dumps(
    {"episode": "e7", "type": "episode", "runner": "worktree"},
    sort_keys=True,
)


def _g5(
    tmp_path,
    monkeypatch,
    *,
    raw,
    items,
    deleted,
    outcomes,
    live=None,
    role="curate",
):
    """V21Checks.g5 on a stub run whose stored typed covers hold SPLIT_ID's work-tree procedure on e7; the fake
    run_procedure answers *outcomes[(item, side)]* (side "p" for the parent tree, "c" for the candidate's).
    """
    v = _checks(
        tmp_path,
        BEFORE,
        AFTER,
        raw,
        items=items,
        deleted=deleted,
        role=role,
        live=live,
    )
    seen = []

    def fake(item, cover, *, tree, **kw):
        side = "p" if tree == v.run.p_tree else "c"
        seen.append((item, side))
        return Outcome(outcomes.get((item, side), False))

    monkeypatch.setattr(gv, "run_procedure", fake)
    v.gate.ev.typed_covers = lambda: [(SPLIT_ID, "e7", COVER)]
    v.gate.ev.signals_for = lambda e: []
    v.gate.python, v.gate._blob = "/usr/bin/python3", lambda s: b""
    from dataclasses import replace

    v.cfg = v.gate.v21 = replace(
        v.cfg,
        episodes=lambda eid: SimpleNamespace(episode_id=eid),
    )
    v.run.c_bodies = {TOKENS: ("function", "x")}
    v.g5()
    return v, seen


def test_retiring_a_procedure_no_remaining_item_reproduces_is_refused(
    tmp_path,
    monkeypatch,
):
    raw = {"why": WHY, "retired": {SPLIT_ID: "superseded"}}
    v, seen = _g5(
        tmp_path,
        monkeypatch,
        raw=raw,
        items=[_item(item=TOKENS)],
        deleted=[SPLIT_ID],
        outcomes={(SPLIT_ID, "p"): True},
    )
    (fail,) = [f for f in v.run.fails if f[0] == "G5"]
    assert (
        fail[2] == SPLIT_ID
        and "reproduced by no remaining item" in fail[1]
        and "e7" in fail[1]
    )
    assert (
        seen == []
    )  # no remaining item lists or forwards the cover: nothing to re-run


def test_aliasing_a_procedure_to_an_equivalent_one_lands(tmp_path, monkeypatch):
    v, seen = _g5(
        tmp_path,
        monkeypatch,
        raw=OK,
        items=[_item(item=TOKENS)],
        deleted=[SPLIT_ID],
        outcomes={(SPLIT_ID, "p"): True, (TOKENS, "c"): True},
    )
    assert [f for f in v.run.fails if f[0] == "G5"] == []
    assert seen == [
        (SPLIT_ID, "p"),
        (TOKENS, "c"),
    ]  # held by the alias target's re-run on the candidate


def test_an_alias_whose_target_answers_differently_is_refused(tmp_path, monkeypatch):
    v, _ = _g5(
        tmp_path,
        monkeypatch,
        raw=OK,
        items=[_item(item=TOKENS)],
        deleted=[SPLIT_ID],
        outcomes={(SPLIT_ID, "p"): True, (TOKENS, "c"): False},
    )
    assert [f[2] for f in v.run.fails if f[0] == "G5"] == [SPLIT_ID]


def test_a_listed_cover_is_held_only_by_its_re_run(tmp_path, monkeypatch):
    from unify.memory_v2.procedures import parse_cover

    listed = _item(item=TOKENS, typed_covers=[parse_cover(json.loads(COVER))])
    raw = {"why": WHY, "retired": {SPLIT_ID: "superseded"}}
    v, seen = _g5(
        tmp_path,
        monkeypatch,
        raw=raw,
        items=[listed],
        deleted=[SPLIT_ID],
        outcomes={(SPLIT_ID, "p"): True, (TOKENS, "c"): False},
    )
    assert [f[2] for f in v.run.fails if f[0] == "G5"] == [
        SPLIT_ID,
    ]  # listing it is no longer enough (Amendment B)
    assert (TOKENS, "c") in seen


# --- RUNTIME's review: R5 (generalise and regroup are CURATE's) and S3 --------------------------------------

GEN_P = {
    "memory/m/__init__.py": '"""M."""\n',
    "memory/m/one.py": '"""One."""\n\n\ndef a(x):\n    """Add one."""\n    return x + 1\n',
    "memory/m/two.py": '"""Two."""\n\n\ndef b(x):\n    """Add one."""\n    return x + 1\n',
}
GEN_C = {
    "memory/m/__init__.py": '"""M."""\n',
    "memory/m/gen.py": '"""Gen."""\n\n\ndef g(x, k=1):\n    """Add k."""\n    return x + k\n',
    "memory/m/one.py": '"""One: a is kept as an alias."""\n\nfrom memory.m.gen import g as a\n',
    "memory/m/two.py": '"""Two: b is kept as an alias."""\n\nfrom memory.m.gen import g as b\n',
}
A, B, G = "memory.m.one:a", "memory.m.two:b", "memory.m.gen:g"


def test_generalising_two_functions_into_one_lands(tmp_path):
    raw = {"why": "a and b are one parametric function", "aliases": {A: G, B: G}}
    v = _checks(tmp_path, GEN_P, GEN_C, raw, items=[_item(item=G)], deleted=[A, B])
    v.curate()
    assert v.run.fails == []


def test_a_new_function_that_no_alias_targets_is_refused(tmp_path):
    cand = {
        **GEN_P,
        "memory/m/two.py": '"""Two."""\n\n\ndef h(x):\n    """Add two."""\n    return x + 2\n',
    }
    raw = {
        "why": "b is wrong; h does it right",
        "retired": {B: "it added the wrong amount"},
    }
    v = _checks(
        tmp_path,
        GEN_P,
        cand,
        raw,
        items=[_item(item="memory.m.two:h")],
        deleted=[B],
    )
    v.curate()
    assert [(c, i) for c, _, i in v.run.fails] == [("G5", "memory.m.two:h")]
    assert "is a new function" in v.run.fails[0][1]


def test_an_alias_to_a_deleted_function_is_refused(tmp_path):
    v = _merge(tmp_path, deleted=[SPLIT_ID, TOKENS])
    assert any(
        f"alias {SPLIT_ID} -> {TOKENS}: the candidate holds no function" in r
        for r in _reasons(v)
    )


def test_one_more_pass_walks_back_over_status_commits(tmp_path, monkeypatch):
    from unify.memory_v2 import memory_writer

    before, after = {**LIB, SPLIT_PATH: SPLIT_ALIAS}, dict(LIB)
    grand = "g" * 40
    young = {SPLIT_ID: {"target": TOKENS, "pass_id": "c0", "commit": grand}}
    monkeypatch.setattr(memory_writer, "is_status_commit", lambda mem, c: c == PARENT)
    v = _checks(tmp_path, before, after, {"why": "drop the old name"}, live=young)
    v.gate.mem = SimpleNamespace(
        run=lambda *a: grand + "\n",
    )  # PARENT^ is the alias's commit
    v.curate()
    assert [c for c, _, _ in v.run.fails] == [
        "G5",
    ] and "stays for one more pass" in v.run.fails[0][1]


def test_a_write_pass_may_not_drop_a_live_alias(tmp_path):
    before, after = {**LIB, SPLIT_PATH: SPLIT_ALIAS}, dict(LIB)
    live = {SPLIT_ID: {"target": TOKENS, "pass_id": "c0", "commit": "o" * 40}}
    v = _checks(tmp_path, before, after, {}, role="write", live=live)
    v.curate()
    assert [c for c, _, _ in v.run.fails] == [
        "G5",
    ] and "may not drop an alias" in v.run.fails[0][1]
