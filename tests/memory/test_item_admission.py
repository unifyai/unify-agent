"""Stage 7: per-item admission (one bad item no longer refuses the pass)."""

import ast
import json
import sqlite3

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.manifest import parse_manifest
from unify.memory.reduction import build, with_dependents
from unify.memory.snapshot import listing, materialise
from tests.memory.test_episodes import _ep
from tests.memory.test_gate import KIT
from tests.memory.test_structural_effect import (
    _fn,
    _item,
    _lookup,
    _test,
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
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=_lookup, library_mode=LibraryConfig())
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


# --- structural Effect, item by item --------------------------------------------------------------------------------------


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


# --- per-item admission with the other item-scoped checks (memory-int2) ---------------------------------

# Replaces the balance it read with a constant under a condition: the override rule (G2) needs covers from
# two episodes, and this one has one.
CAPPED = _fn(
    "capped",
    'v = apis.venmo.balance()["balance"]\nif v > 100:\n    v = 0\nreturn v',
)
T_CAPPED = _test("capped", "balance", {"balance": 3}, 3)
