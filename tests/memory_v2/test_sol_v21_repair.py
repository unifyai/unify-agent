import asyncio
import json
import time
from decimal import Decimal

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.drafts import draft_states
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate, GateResult
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sol_pass import PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_gate import ITEM, KIT, MAN, MOD, TEST
from tests.memory_v2.test_sol_pass import (
    Turns,
    _call,
    _e1,
    _ME,
    _run,
    _sol,
    _write,
    needs_bwrap,
)

# the function reads a key the recorded response does not have: its test fails on the candidate (G3)
MOD_BAD = MOD.replace('["user_id"]', '["user"]')
FILES_CODE = "\n".join(
    [
        _write("/memory/unify_memory_testkit.py", KIT),
        _write("/memory/env/venmo/tests/test_me.py", TEST),
        _write("/memory/env/venmo/__init__.py", MOD),
        _write("/memory/.pass/manifest.json", json.dumps({**MAN, "summary": "s"})),
    ],
)


def _count(gate, seen=None):
    """Count the pass's gate calls, passing each through to the real gate; *seen* collects (name, candidate,
    seed) per call."""
    calls = {"check": 0, "merge": 0}
    for name in ("check", "merge"):
        real = getattr(gate, name)

        def wrapped(*args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            if seen is not None:
                seen.append((_name, args[1], kwargs.get("seed")))
            return _real(*args, **kwargs)

        setattr(gate, name, wrapped)
    return calls


@needs_bwrap
def test_repair_round_lands_one_commit_on_the_parent(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    seen: list = []
    calls = _count(gate, seen)
    manifest = {**MAN, "items": [ITEM], "summary": "s"}
    first_try = "\n".join(
        [
            _write("/memory/unify_memory_testkit.py", KIT),
            _write("/memory/env/venmo/tests/test_me.py", TEST),
            _write("/memory/env/venmo/__init__.py", MOD_BAD),
            _write("/memory/.pass/manifest.json", json.dumps(manifest)),
        ],
    )
    model = Turns(
        [
            _call(
                "dm",
                "dismiss",
                {
                    "episode": "e1",
                    "reason": "one recorded call; the item below covers it",
                },
            ),
            _call("w0", "execute_code", {"code": first_try}),
            _call("f0", "finish", {"summary": "first try"}),
            _call("r1", "read", {"path": "/inputs/gate/result-0.md"}),
            _call(
                "w1",
                "execute_code",
                {"code": _write("/memory/env/venmo/__init__.py", MOD)},
            ),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(max_calls=20, v21=True),
    )
    parent = mem.head()
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    assert out.passed, out.reasons
    assert out.rounds == 2 and out.round_results == ["/inputs/gate/result-0.md"]
    # round 0's check refused; round 1's check passed, so its candidate went to the one merge
    assert calls == {"check": 2, "merge": 1}
    result = model.outputs[
        "r1"
    ]  # what Sol read: the first 8,000 bytes, marked when there is more
    assert result.startswith(
        "# Gate result: pass p1, round 0, check (not merged)",
    ), result
    repair_msgs = [
        m
        for m in model.seen
        if m.get("role") == "user" and "repair round 1 of 2" in m["content"]
    ]
    assert (
        len(repair_msgs) == 1
        and "/inputs/gate/result-0.md" in repair_msgs[0]["content"]
    )
    # one commit on the parent: the refused round never enters main's history (D13)
    assert mem.head() == out.commit
    assert mem.run("rev-list", "--count", f"{parent}..{out.commit}").strip() == "1"
    assert "Round:" not in mem.run("log", "-1", "--format=%B", out.commit)
    # Amendment A: one seed for the pass (its first candidate's), and the merge lands the last checked commit
    from unify.memory_v2.qa import seed_of

    c0, c1, m = seen
    assert [n for n, _, _ in seen] == ["check", "check", "merge"]
    assert c0[2] == c1[2] == m[2] == seed_of(c0[1])
    assert m[1] == c1[1] == out.commit
    (row,) = ev.pass_rounds()
    assert (
        row["pass_id"] == "p1" and row["rounds"] == 2 and len(row["round_blobs"]) == 1
    )
    assert (
        row["patch_blob"] is None and draft_states(ev.pass_rounds()) == {}
    )  # landed whole: no draft
    full = sol.gate.blobs.get(
        row["round_blobs"][0],
    ).decode()  # the same text, kept whole
    assert full.startswith(result.split("\n[… shown bytes ")[0])
    assert "G3: env/venmo/tests/test_me.py is not green on the candidate" in full
    # the failing test's own output, not only the reason's 300-character tail
    assert "## Test output" in full and "def test_me():" in full and "KeyError" in full


@needs_bwrap
def test_budget_cut_skips_the_check_and_merges_once(tmp_path):
    model = Turns(
        [
            _call("w", "execute_code", {"code": FILES_CODE}),
            _call("f", "finish", {"summary": "s"}),
        ],
        usd="0.30",
    )
    mem, ev, sol = _sol(tmp_path, model, v21=True, max_usd=Decimal("0.70"))
    calls = _count(sol.gate)
    out = _run(sol)
    # round 0 cost 0.60 of the 0.70 cap: the 0.10 left cannot hold a round, so no check runs
    assert calls == {"check": 0, "merge": 1}
    assert out.rounds == 1 and out.round_results == []
    assert (
        "repair: no round 1: 0.10 USD left, below the round reserve of 0.60 USD"
        in out.reasons
    )
    # the gate (no action lookup here) refuses: per-item admission ran in the one merge, and the patch is a draft
    assert not out.passed
    (row,) = ev.pass_rounds()
    assert row["rounds"] == 1 and row["patch_blob"] is not None
    assert draft_states(ev.pass_rounds()) == {"px": "open"}
    final = sol.gate.blobs.get(row["gate_blob"]).decode()
    assert final.startswith("# Gate result: pass px, round 0, final (merge)")
    assert (
        "repair: no round 1: 0.10 USD left" in final
    )  # the pass's own notes are in the final result


@needs_bwrap
def test_gate_time_counts_against_the_deadline(tmp_path):
    model = Turns(
        [
            _call("w", "execute_code", {"code": FILES_CODE}),
            _call("f", "finish", {"summary": "s"}),
        ],
    )
    mem, ev, sol = _sol(tmp_path, model, v21=True, deadline_s=6.0)
    seen = []

    def slow_check(parent, candidate, manifest, **kwargs):
        seen.append(candidate)
        time.sleep(4.0)  # the jail's time: the gate's test runs
        return GateResult(
            False,
            {"G2": False},
            ["G2: env/venmo:me covers no recorded action"],
            ["G2"],
            items_refused={"env/venmo:me": ["G2"]},
        )

    sol.gate.check = slow_check
    out = _run(sol)
    # the check ran once and was written for Sol; after it, less time was left than a round takes
    assert len(seen) == 1 and out.rounds == 1
    assert out.round_results == ["/inputs/gate/result-0.md"]
    assert any(
        r.startswith("repair: no round 1: ") and "below the round time reserve" in r
        for r in out.reasons
    ), out.reasons
    # the round's candidate is one commit on the parent that no ref names
    assert mem.run("rev-list", "--count", f"{mem.head()}..{seen[0]}").strip() == "1"
    assert mem.run("branch", "--contains", seen[0]).strip() == ""
    assert "Round:" not in mem.run(
        "log",
        "-1",
        "--format=%B",
        seen[0],
    )  # Amendment A: no round trailer at all


@needs_bwrap
def test_check_runs_the_items_own_tests_under_v21(tmp_path):
    """Amendment B: under v2.1 ``check`` also runs the items' own tests (gate.check, no merge): a failing test's
    output comes back as the head of the gate's full result."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    calls = _count(gate)
    manifest = {**MAN, "items": [ITEM], "summary": "s"}
    bad = "\n".join(
        [
            _write("/memory/unify_memory_testkit.py", KIT),
            _write("/memory/env/venmo/tests/test_me.py", TEST),
            _write("/memory/env/venmo/__init__.py", MOD_BAD),
            _write("/memory/.pass/manifest.json", json.dumps(manifest)),
        ],
    )
    model = Turns(
        [
            _call(
                "dm",
                "dismiss",
                {
                    "episode": "e1",
                    "reason": "one recorded call; the item below covers it",
                },
            ),
            _call("w0", "execute_code", {"code": bad}),
            _call("c0", "check", {"manifest": json.dumps(manifest)}),
            _call(
                "w1",
                "execute_code",
                {"code": _write("/memory/env/venmo/__init__.py", MOD)},
            ),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(max_calls=20, v21=True),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    reply = model.outputs["c0"]
    assert reply.startswith(
        "# Gate result: pass p1, round 0, check (not merged)",
    ), reply
    assert "is not green on the candidate" in reply and "KeyError" in reply
    assert out.passed and calls["merge"] == 1
    assert (
        len(ev.pass_rounds()) == 1
    )  # the check tool wrote nothing of its own to the evidence store


@needs_bwrap
def test_a_long_check_result_is_a_file_read_can_page(tmp_path):
    """P2T4-1: a check refusal over the view bound is written to /inputs/gate/check-<k>.md, so the marker's
    offset can be read; nothing in it is unreachable."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    long = (
        "G3: env/venmo/tests/test_me.py is not green on the candidate "
        + "w" * 9000
        + " THE-END"
    )
    gate.check = lambda parent, candidate, manifest, **k: GateResult(
        False,
        {"G3": False},
        [long],
        ["G3"],
        items_refused={"env/venmo:me": ["G3"]},
    )
    manifest = {**MAN, "items": [ITEM], "summary": "s"}
    model = Turns(
        [
            _call(
                "dm",
                "dismiss",
                {
                    "episode": "e1",
                    "reason": "one recorded call; the item below covers it",
                },
            ),
            _call("w0", "execute_code", {"code": FILES_CODE}),
            _call("c0", "check", {"manifest": json.dumps(manifest)}),
            _call("r0", "read", {"path": "/inputs/gate/check-0.md", "offset": 8000}),
        ],
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(max_calls=20, v21=True),
    )
    asyncio.run(sol.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    reply = model.outputs["c0"]
    assert reply.startswith("# Gate result: pass p1, round 0, check (not merged)")
    assert "next: offset=8000]" in reply and reply.endswith(
        "(full result: /inputs/gate/check-0.md)",
    )
    assert (
        "THE-END" not in reply and "THE-END" in model.outputs["r0"]
    )  # the rest is reachable by read
