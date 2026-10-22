"""Stage 7: a function's ``write`` effect derived from the calls it covers, and writers marked in the index."""

import shutil

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate, _covers_a_write
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.index import build_index
from tests.memory.test_episodes import _ep

sandboxed = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

ACTS = {
    0: Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read"),
    3: Action(3, "venmo", "balance", [], {}, {"balance": 3}, "ok", "read"),
    4: Action(4, "venmo", "friends", [], {}, {"friends": ["f"]}, "ok", "read"),
    5: Action(5, "venmo", "pay", [], {}, {"ok": True}, "ok", "write"),
}


def _lookup(eid, i):
    return ACTS.get(i) if eid == "e1" else None


def _fn(name, body, effect="read", form="env"):
    return (
        f"\n\ndef {name}(apis):\n"
        f'    """{name.capitalize()} from the venmo record.\n\n'
        f"    Effect: {effect}\n    Input: {form}\n"
        f'    """\n' + "".join(f"    {line}\n" for line in body.splitlines())
    )


def _test(name, method, response, expect, extra=""):
    return (
        "from unify_memory_testkit import env_from\n"
        f"from env.venmo import {name}\n\n"
        f"def test_{name}():\n"
        f"    assert {name}(env_from([('venmo', '{method}', {{}}, {response!r})])) == {expect!r}\n"
        + extra
    )


def _item(name, covers, tests=None):
    return {
        "item": f"env/venmo:{name}",
        "kind": "env_function",
        "source_episodes": ["e1"],
        "tests": tests or [f"env/venmo/tests/test_{name}.py"],
        "covers": covers,
        "input": "env",
    }


ME = _fn("me", 'return apis.venmo.me()["user_id"]')
T_ME = _test("me", "me", {"user_id": "u-1"}, "u-1")
T_PAY = _test("pay", "pay", {"ok": True}, True)


def _pay(effect):
    return _fn("pay", 'return apis.venmo.pay()["ok"]', effect=effect)


@pytest.fixture
def world(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(episode_id="e1"), "1" * 40)
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=_lookup, library_mode=LibraryConfig())
    return mem, ev, gate


def _index_lines(tree):
    return {
        ln.split("`")[1].split("(")[0]: ln
        for ln in build_index(tree).splitlines()
        if ln.startswith("- `")
    }


def test_write_is_read_from_the_recording_never_from_names():
    write = Action(5, "venmo", "list_things", [], {}, {"ok": True}, "ok", "write")
    named = Action(6, "venmo", "create_payment", [], {}, {"ok": True}, "ok", "unknown")
    read = Action(7, "venmo", "delete_cache", [], {}, {"ok": True}, "ok", "read")
    assert _covers_a_write([("e", 5, write)], "env")
    assert _covers_a_write([("e", 5, write)], None)
    assert not _covers_a_write([("e", 6, named), ("e", 7, read)], "env")
    # a parser of a recorded response makes no call
    assert not _covers_a_write([("e", 5, write)], "observation")
    # a file write is an observation the function reads
    wt = Action(
        8,
        "worktree:workspace",
        "write",
        ["a.csv"],
        {},
        {},
        "ok",
        "write",
        kind="worktree",
    )
    assert not _covers_a_write([("e", 8, wt)], "path")
    assert not _covers_a_write([("e", 8, wt)], None)
