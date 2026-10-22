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
