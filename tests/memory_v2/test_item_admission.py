"""Stage 7: per-item admission (one bad item no longer refuses the pass)."""

import ast
import json
import sqlite3

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.manifest import parse_manifest
from unify.memory_v2.reduction import build, with_dependents
from unify.memory_v2.snapshot import listing, materialise
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import KIT, _candidate, _merged
from tests.memory_v2.test_structural_effect import (
    _fn,
    _item,
    _lookup,
    _test,
    sandboxed,
)

ME = _fn("me", 'return apis.venmo.me()["user_id"]')
BALANCE = _fn("balance", 'return apis.venmo.balance()["balance"]')
FRIENDS = _fn("friends", 'return apis.venmo.friends()["friends"]')
T_ME = _test("me", "me", {"user_id": "u-1"}, "u-1")
T_BALANCE = _test("balance", "balance", {"balance": 3}, 3)
T_FRIENDS = _test("friends", "friends", {"friends": ["f"]}, ["f"])
THREE = {
    "env/venmo/__init__.py": '"""Venmo."""' + ME + BALANCE + FRIENDS,
    "env/venmo/tests/test_me.py": T_ME,
    "env/venmo/tests/test_balance.py": T_BALANCE,
    "env/venmo/tests/test_friends.py": T_FRIENDS,
    "unify_memory_testkit.py": KIT,
}
# friends cites an action that was never recorded: an item-scoped G2 failure
THREE_MAN = {
    "items": [
        _item("me", [["e1", 0]]),
        _item("balance", [["e1", 3]]),
        _item("friends", [["e1", 9]]),
    ],
    "support": ["unify_memory_testkit.py"],
    "skeleton": ["env/venmo"],
}


@pytest.fixture
def world(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="e1"), "1" * 40)
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=_lookup)
    return mem, ev, gate


def _tree(mem, sha, dest):
    files, _ = listing(mem, sha)
    return files, materialise(mem, files, dest)


def _defs(source):
    return {
        n.name: ast.unparse(n)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
    }


def _row(ev, pass_id):
    passed, merged, items_merged, items_refused, patch = ev.db.execute(
        "SELECT passed, merged, items_merged, items_refused, patch_blob FROM passes "
        "WHERE pass_id=?",
        (pass_id,),
    ).fetchone()
    return passed, merged, json.loads(items_merged), json.loads(items_refused), patch


# --- per-item admission through the gate ------------------------------------------------------------------


@sandboxed
def test_one_item_failing_g2_is_refused_and_the_other_two_merge(world, tmp_path):
    mem, ev, gate = world
    parent = mem.head()
    cand = _candidate(mem, THREE)
    res = gate.merge(parent, cand, THREE_MAN, "p1", "incremental", "venmo", "0.01")
    assert res.passed, res.reasons
    assert res.items_merged == ["env/venmo:me", "env/venmo:balance"]
    assert res.items_refused == {"env/venmo:friends": ["G2"]}
    assert mem.head() == res.merged != cand
    # the reduction is a child of the parent: main never holds the refused item, even in its history
    assert mem.run("rev-parse", f"{res.merged}^").strip() == parent
    files, tree = _tree(mem, res.merged, tmp_path / "landed")
    assert set(_defs((tree / "env/venmo/__init__.py").read_text())) == {"me", "balance"}
    assert "env/venmo/tests/test_friends.py" not in files
    assert ev.covered() == {("e1", 0), ("e1", 3)}
    assert any(r.startswith("item refused: G2: env/venmo:friends") for r in res.reasons)
    passed, merged, items_merged, items_refused, patch = _row(ev, "p1")
    assert (passed, merged) == (1, res.merged)
    assert items_merged == res.items_merged and items_refused == res.items_refused
    assert patch is not None  # the refused part is kept
    # reason codes only: no free text in the per-item record
    assert all(
        c in ("G1", "G2", "G3", "G6", "dependency", "pass")
        for codes in items_refused.values()
        for c in codes
    )


@sandboxed
def test_shared_module_a_refused_change_gets_the_parents_def_back(world, tmp_path):
    mem, ev, gate = world
    _merged(
        mem,
        {
            "env/venmo/__init__.py": '"""Venmo."""' + ME,
            "env/venmo/tests/test_me.py": T_ME,
            "unify_memory_testkit.py": KIT,
        },
    )
    parent = mem.head()
    me2 = _fn(
        "me",
        "r = apis.venmo.me()\n"
        'if not isinstance(r, dict) or "user_id" not in r:\n'
        '    raise ValueError("me: the response has no user_id")\n'
        'return r["user_id"]',
    )
    t_me2 = T_ME + (
        "\n\ndef test_me_shape():\n"
        "    import pytest\n"
        "    with pytest.raises(ValueError):\n"
        "        me(env_from([('venmo', 'me', {}, {})]))\n"
    )
    cand = _candidate(
        mem,
        {
            "env/venmo/__init__.py": '"""Venmo."""' + me2 + BALANCE,
            "env/venmo/tests/test_me.py": t_me2,
            "env/venmo/tests/test_balance.py": T_BALANCE,
        },
    )
    man = {
        "items": [
            _item("me", [["e1", 0], ["e1", 9]]),  # one fabricated cover
            _item("balance", [["e1", 3]]),
        ],
    }
    res = gate.merge(parent, cand, man, "p2", "incremental", "venmo", "0")
    assert res.passed, res.reasons
    assert res.items_merged == ["env/venmo:balance"]
    assert res.items_refused == {"env/venmo:me": ["G2"]}
    _, before = _tree(mem, parent, tmp_path / "before")
    _, after = _tree(mem, res.merged, tmp_path / "after")
    old = _defs((before / "env/venmo/__init__.py").read_text())
    new = _defs((after / "env/venmo/__init__.py").read_text())
    assert new["me"] == old["me"] and "balance" in new
    assert (after / "env/venmo/tests/test_me.py").read_text() == T_ME


@sandboxed
def test_an_item_that_calls_a_refused_item_is_refused_with_it(world, tmp_path):
    mem, ev, gate = world
    parent = mem.head()
    rich = _fn("rich", "return balance(apis) > 2")
    files = {
        "env/venmo/__init__.py": '"""Venmo."""' + ME + BALANCE + rich,
        "env/venmo/tests/test_me.py": T_ME,
        "env/venmo/tests/test_balance.py": T_BALANCE,
        "env/venmo/tests/test_rich.py": _test("rich", "balance", {"balance": 3}, True),
        "unify_memory_testkit.py": KIT,
    }
    man = {
        "items": [
            _item("me", [["e1", 0]]),
            _item("balance", [["e1", 9]]),  # refused: G2
            _item("rich", [["e1", 3]]),  # calls balance
        ],
        "support": ["unify_memory_testkit.py"],
        "skeleton": ["env/venmo"],
    }
    res = gate.merge(
        parent,
        _candidate(mem, files),
        man,
        "p3",
        "incremental",
        "venmo",
        "0",
    )
    assert res.passed, res.reasons
    assert res.items_merged == ["env/venmo:me"]
    assert res.items_refused == {
        "env/venmo:balance": ["G2"],
        "env/venmo:rich": ["dependency"],
    }
    assert any(
        "env/venmo:rich is refused with env/venmo:balance" in r for r in res.reasons
    )
    _, tree = _tree(mem, res.merged, tmp_path / "landed")
    assert set(_defs((tree / "env/venmo/__init__.py").read_text())) == {"me"}


@sandboxed
def test_a_pass_wide_failure_still_refuses_every_item(world):
    mem, ev, gate = world
    parent = mem.head()
    # an undeclared data file is pass-wide (G1), beside the item-scoped G2 failure
    cand = _candidate(mem, {**THREE, "env/venmo/tests/extra.json": "{}"})
    res = gate.merge(parent, cand, THREE_MAN, "p4", "incremental", "venmo", "0")
    assert not res.passed and mem.head() == parent and res.merged is None
    assert res.items_merged == []
    assert res.items_refused == {
        "env/venmo:me": ["pass"],
        "env/venmo:balance": ["pass"],
        "env/venmo:friends": ["G2"],
    }
    assert not any(r.startswith("item refused") for r in res.reasons)
    passed, merged, items_merged, items_refused, patch = _row(ev, "p4")
    assert (passed, merged, items_merged) == (0, None, [])
    assert items_refused == res.items_refused and patch is not None
    assert ev.covered() == set()


@sandboxed
def test_a_reduction_the_gate_refuses_leaves_the_pass_refused_whole(world):
    mem, ev, gate = world
    parent = mem.head()
    # me's test reaches friends through the whole channel, which no reduction can see: one round, then refuse
    t_me = (
        "import env.venmo as v\nfrom unify_memory_testkit import env_from\n\n"
        "def test_me():\n"
        "    assert v.me(env_from([('venmo', 'me', {}, {'user_id': 'u-1'})])) == 'u-1'\n"
        "    assert hasattr(v, 'friends')\n"
    )
    cand = _candidate(mem, {**THREE, "env/venmo/tests/test_me.py": t_me})
    res = gate.merge(parent, cand, THREE_MAN, "p5", "incremental", "venmo", "0")
    assert not res.passed and mem.head() == parent
    assert any("was refused too" in r for r in res.reasons)
    assert res.items_merged == [] and res.items_refused["env/venmo:friends"] == ["G2"]
    assert ev.covered() == set()


# --- structural Effect, item by item --------------------------------------------------------------------------------------


@sandboxed
def test_an_effect_mismatch_is_refused_alone_and_its_sibling_lands(world):
    mem, ev, gate = world
    parent = mem.head()
    files = {
        "env/venmo/__init__.py": '"""Venmo."""'
        + ME
        + _fn("pay", 'return apis.venmo.pay()["ok"]', effect="read"),
        "env/venmo/tests/test_me.py": T_ME,
        "env/venmo/tests/test_pay.py": _test("pay", "pay", {"ok": True}, True),
        "unify_memory_testkit.py": KIT,
    }
    man = {
        "items": [_item("me", [["e1", 0]]), _item("pay", [["e1", 5]])],
        "support": ["unify_memory_testkit.py"],
        "skeleton": ["env/venmo"],
    }
    res = gate.merge(
        parent,
        _candidate(mem, files),
        man,
        "e1p",
        "incremental",
        "venmo",
        "0",
    )
    assert res.passed, res.reasons
    assert res.items_merged == ["env/venmo:me"]
    assert res.items_refused == {"env/venmo:pay": ["G6"]}
    assert ev.covered() == {("e1", 0)}


# --- the reduction itself (no sandbox) ----------------------------------------------------------------------


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_reduction_removes_a_new_def_keeps_its_siblings_and_restores_notes(tmp_path):
    parent = _write(
        tmp_path / "p",
        {
            "env/venmo/__init__.py": '"""Venmo."""\n__all__ = ["me"]\n' + ME,
            "env/venmo/NOTES.md": "# venmo\n\n## Paging\nold\n",
        },
    )
    helper = "\n\ndef _shape(r):\n    return r\n"
    cand = _write(
        tmp_path / "c",
        {
            "env/venmo/__init__.py": '"""Venmo."""\n__all__ = ["me", "balance", "friends"]\n'
            + helper
            + ME
            + _fn("balance", 'return _shape(apis.venmo.balance())["balance"]')
            + FRIENDS,
            "env/venmo/NOTES.md": "# venmo\n\n## Paging\nnew\n\n## Auth\nonce\n",
            "env/venmo/tests/test_balance.py": T_BALANCE,
            "env/venmo/tests/test_friends.py": T_FRIENDS,
        },
    )
    raw = {
        "items": [
            _item("balance", [["e1", 3]]),
            _item("friends", [["e1", 4]]),
            {
                "item": "env/venmo/NOTES.md#paging",
                "kind": "env_note",
                "source_episodes": ["e1"],
            },
            {
                "item": "env/venmo/NOTES.md#auth",
                "kind": "env_note",
                "source_episodes": ["e1"],
            },
        ],
        "skeleton": ["env/venmo"],
    }
    man = parse_manifest(raw)
    refused = {"env/venmo:balance": ["G2"], "env/venmo/NOTES.md#paging": ["G1"]}
    out = build(
        man,
        raw,
        refused,
        {"env/venmo/__init__.py", "env/venmo/NOTES.md"},
        {"env/venmo:me"},
        parent,
        cand,
        tmp_path / "r",
    )
    red = tmp_path / "r"
    mod = (red / "env/venmo/__init__.py").read_text()
    assert set(_defs(mod)) == {
        "_shape",
        "me",
        "friends",
    }  # the helper only balance used stays
    assert '"balance"' not in mod.split("\n", 2)[1]  # nor is it in __all__
    assert (
        red / "env/venmo/NOTES.md"
    ).read_text() == "# venmo\n\n## Paging\nold\n## Auth\nonce\n"
    assert not (red / "env/venmo/tests/test_balance.py").exists()
    assert (red / "env/venmo/tests/test_friends.py").exists()
    assert [i["item"] for i in out["items"]] == [
        "env/venmo:friends",
        "env/venmo/NOTES.md#auth",
    ]
    assert out["skeleton"] == ["env/venmo"]


def test_reduction_removes_a_new_channel_no_admitted_item_uses(tmp_path):
    parent = _write(tmp_path / "p", {"env/venmo/__init__.py": ME})
    cand = _write(
        tmp_path / "c",
        {
            "env/venmo/__init__.py": ME + BALANCE,
            "env/venmo/tests/test_balance.py": T_BALANCE,
            "env/slack/__init__.py": '"""Slack."""' + _fn("post", "return 1"),
            "env/slack/tests/test_post.py": "def test_post():\n    pass\n",
            "env/slack/tests/data.json": "{}",
        },
    )
    post = {**_item("post", [["e1", 2]]), "item": "env/slack:post"}
    post["tests"] = ["env/slack/tests/test_post.py"]
    raw = {
        "items": [_item("balance", [["e1", 3]]), post],
        "skeleton": ["env/slack"],
        "support": ["env/slack/tests/data.json"],
    }
    out = build(
        parse_manifest(raw),
        raw,
        {"env/slack:post": ["G2"]},
        {"env/venmo/__init__.py"},
        {"env/venmo:me"},
        parent,
        cand,
        tmp_path / "r",
    )
    assert not (tmp_path / "r/env/slack").exists()
    assert out["skeleton"] == [] and out["support"] == []
    assert [i["item"] for i in out["items"]] == ["env/venmo:balance"]


def test_reduction_refuses_dependents_by_call_test_import_and_shared_test(tmp_path):
    cand = _write(
        tmp_path / "c",
        {
            "env/venmo/__init__.py": ME
            + BALANCE
            + "\n\ndef _via(apis):\n    return balance(apis)\n"
            + _fn("rich", "return _via(apis) > 2")
            + FRIENDS
            + _fn("pay", "return 1"),
            "env/venmo/tests/test_friends.py": "from env.venmo import friends, me\n",
            "env/venmo/tests/test_shared.py": "",
        },
    )
    raw = {
        "items": [
            _item("me", [["e1", 0]]),
            _item("balance", [["e1", 3]], ["env/venmo/tests/test_shared.py"]),
            _item("rich", [["e1", 3]]),
            _item("friends", [["e1", 4]]),
            _item("pay", [["e1", 5]], ["env/venmo/tests/test_shared.py"]),
        ],
    }
    refused, reasons = with_dependents(
        parse_manifest(raw),
        {"env/venmo:me": ["G2"], "env/venmo:balance": ["G3"]},
        cand,
    )
    assert refused == {
        "env/venmo:me": ["G2"],
        "env/venmo:balance": ["G3"],
        "env/venmo:rich": ["dependency"],  # through the private helper
        "env/venmo:friends": ["dependency"],  # its test imports me
        "env/venmo:pay": ["dependency"],  # it shares balance's test file
    }
    assert len(reasons) == 3


def test_no_reduction_when_nothing_is_left_or_a_skeleton_change_holds_a_refused_change(
    tmp_path,
):
    parent = _write(tmp_path / "p", {"env/venmo/__init__.py": ME})
    cand = _write(tmp_path / "c", {"env/venmo/__init__.py": '"""V."""' + ME + BALANCE})
    raw = {"items": [_item("me", [["e1", 0]]), _item("balance", [["e1", 3]])]}
    man = parse_manifest(raw)
    args = ({"env/venmo/__init__.py"}, {"env/venmo:me"}, parent, cand)
    both = {"env/venmo:me": ["G2"], "env/venmo:balance": ["G2"]}
    assert build(man, raw, both, *args, tmp_path / "r1") is None
    skel = {**raw, "skeleton": ["env/venmo"]}
    assert (
        build(
            parse_manifest(skel),
            skel,
            {"env/venmo:me": ["G2"]},
            *args,
            tmp_path / "r2",
        )
        is None
    )


def test_an_old_evidence_store_gains_the_per_item_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(str(path))
    db.execute(
        "CREATE TABLE passes(pass_id TEXT PRIMARY KEY, kind TEXT, channel TEXT, parent TEXT, "
        "candidate TEXT, passed INTEGER, reasons TEXT, usd TEXT, patch_blob TEXT)",
    )
    db.execute("INSERT INTO passes VALUES('old','k',NULL,'p','c',1,'[]','0',NULL)")
    db.commit()
    db.close()
    ev = EvidenceStore(path)
    ev.record_pass(
        {
            "pass_id": "new",
            "passed": 1,
            "merged": "m",
            "items_merged": '["env/venmo:me"]',
            "items_refused": "{}",
        },
    )
    rows = ev.db.execute(
        "SELECT pass_id, merged, items_merged FROM passes ORDER BY pass_id",
    ).fetchall()
    assert rows == [("new", "m", '["env/venmo:me"]'), ("old", None, None)]


# --- per-item admission with the other item-scoped checks (memory-v2-int2) ---------------------------------

# Replaces the balance it read with a constant under a condition: the override rule (G2) needs covers from
# two episodes, and this one has one.
CAPPED = _fn(
    "capped",
    'v = apis.venmo.balance()["balance"]\nif v > 100:\n    v = 0\nreturn v',
)
T_CAPPED = _test("capped", "balance", {"balance": 3}, 3)


@sandboxed
def test_a_function_the_rule_check_refuses_is_refused_alone(world):
    mem, ev, gate = world
    files = {
        "env/venmo/__init__.py": '"""Venmo."""' + ME + CAPPED,
        "env/venmo/tests/test_me.py": T_ME,
        "env/venmo/tests/test_capped.py": T_CAPPED,
        "unify_memory_testkit.py": KIT,
    }
    man = {
        "items": [_item("me", [["e1", 0]]), _item("capped", [["e1", 3]])],
        "support": ["unify_memory_testkit.py"],
        "skeleton": ["env/venmo"],
    }
    res = gate.merge(
        mem.head(),
        _candidate(mem, files),
        man,
        "p-rule",
        "incremental",
        "venmo",
        "0.01",
    )
    assert res.passed, res.reasons
    assert res.items_merged == ["env/venmo:me"]
    assert res.items_refused == {"env/venmo:capped": ["G2"]}
    assert any(
        r.startswith("item refused: G2: env/venmo:capped replaces a value")
        for r in res.reasons
    ), res.reasons


def test_stage5_refusals_belong_to_their_items_and_join_the_reduction(tmp_path):
    from types import SimpleNamespace

    from unify.memory_v2.gate import CHECKS, GateResult, _Run
    from unify.memory_v2.qa import QAChecks, QAConfig

    res = GateResult(True, {c: True for c in CHECKS})
    run = _Run(res, parse_manifest(THREE_MAN), THREE_MAN, "p", "c" * 40, tmp_path)
    qa = QAChecks(SimpleNamespace(qa=QAConfig(determinism=True)), run)
    ran = []
    qa._dynamic = lambda: ran.append(1)
    run.qa_env = object()  # the kit is mounted
    run.fail("G2", "env/venmo:friends covers (e1,9), not recorded", "env/venmo:friends")
    qa.dynamic()  # Gate.check: never on a refused candidate
    assert ran == []
    qa.dynamic(item_scoped=True)  # Gate.merge: a candidate refused item by item only
    assert ran == [1]
    qa._fail("mutation", "env/venmo:me's tests kill 0 of 3 mutants", "env/venmo:me")
    test = "env/venmo/tests/test_balance.py"
    qa._fail("determinism", f"{test} gives different outcomes", qa._owners(test))
    assert run.item_fail == {
        "env/venmo:friends": ["G2"],
        "env/venmo:me": ["G3"],
        "env/venmo:balance": ["G3"],
    }
    assert not run.pass_wide
    assert "G3: [qa:mutation] env/venmo:me's tests kill 0 of 3 mutants" in res.reasons
    # a test file no item lists, or the spent budget, belongs to the whole pass
    qa._fail(
        "fixture-size",
        "env/venmo/tests/data.json is too big",
        qa._owners("x.json"),
    )
    assert run.pass_wide
    ran.clear()
    qa.dynamic(item_scoped=True)
    assert ran == []
