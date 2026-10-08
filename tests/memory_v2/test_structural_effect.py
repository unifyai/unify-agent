"""Stage 7: a function's ``write`` effect derived from the calls it covers, and writers marked in the index."""

import shutil

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate, _covers_a_write
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import build_index
from unify.memory_v2.snapshot import listing, materialise
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import KIT, _candidate

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
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=_lookup)
    return mem, ev, gate


def _index_lines(tree):
    return {
        ln.split("`")[1].split("(")[0]: ln
        for ln in build_index(tree).splitlines()
        if ln.startswith("- `")
    }


@sandboxed
def test_a_writer_declared_read_is_refused_and_declared_write_lands_marked(
    world,
    tmp_path,
):
    mem, ev, gate = world
    parent = mem.head()
    man = {"items": [_item("pay", [["e1", 5]])], "support": ["unify_memory_testkit.py"]}
    files = {
        "env/venmo/__init__.py": _pay("read"),
        "env/venmo/tests/test_pay.py": T_PAY,
        "unify_memory_testkit.py": KIT,
    }
    snap = tmp_path / "snap"
    for rel, text in files.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_text(text)
    # the consolidator's check names it before the pass ends
    assert any(
        "env/venmo:pay covers a recorded write call" in r
        for r in gate.preview(parent, snap, man)
    )
    res = gate.merge(
        parent,
        _candidate(mem, files),
        man,
        "w1",
        "incremental",
        "venmo",
        "0",
    )
    assert not res.passed and not res.checks["G6"] and mem.head() == parent
    assert any(
        "env/venmo:pay covers a recorded write call, so it must declare Effect: write"
        in r
        for r in res.reasons
    )
    files["env/venmo/__init__.py"] = _pay("write")
    res = gate.merge(
        parent,
        _candidate(mem, files),
        man,
        "w2",
        "incremental",
        "venmo",
        "0",
    )
    assert res.passed, res.reasons
    lines = _index_lines(
        materialise(mem, listing(mem, mem.head())[0], tmp_path / "landed"),
    )
    assert lines["pay"].endswith("(input: env) (writes)")


@sandboxed
def test_a_parser_of_a_recorded_write_needs_no_write_effect(world):
    mem, ev, gate = world
    parent = mem.head()
    # a parser of the write's recorded response, declared read: it makes no call
    parse = _fn("paid", 'return obs["ok"]', form="observation").replace(
        "(apis)",
        "(obs)",
    )
    files = {
        "env/venmo/__init__.py": parse,
        "env/venmo/tests/test_paid.py": (
            "from env.venmo import paid\n\ndef test_paid():\n    assert paid({'ok': True})\n"
        ),
    }
    man = {"items": [{**_item("paid", [["e1", 5]]), "input": "observation"}]}
    res = gate.merge(
        parent,
        _candidate(mem, files),
        man,
        "w3",
        "incremental",
        "venmo",
        "0",
    )
    assert res.passed, res.reasons


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
