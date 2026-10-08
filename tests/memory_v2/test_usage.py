"""Implicit use signals and the library-use table each consolidation pass's first message carries."""

from __future__ import annotations

import asyncio

from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_use_telemetry import MODULE, _index, _record
from unify.memory_v2 import usage
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.signals import Signal
from unify.memory_v2.sol_pass import PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest


def test_item_signals_flag_used_never_used_and_refusing_accepted_inputs(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    _index(
        ev,
        "e1",
        _record(
            {"env/x:parse": {"called": 1, "refused": 1, "refused_then_accepted": 1}},
            unknown={"*": 1},
        ),
        "2026-10-08T01:00:00Z",
    )
    _index(
        ev,
        "e2",
        _record({"env/x:strict": {"referenced": 1}}),
        "2026-10-08T02:00:00Z",
    )
    parse = usage.item_signals("env/x:parse", ev)
    assert (
        parse["used"] and parse["refusing_accepted_inputs"] and not parse["never_used"]
    )
    assert parse["uncertain"] and parse["last_call_seq"] == ev.seq_of("e1")
    lookup = usage.item_signals("env/x:lookup", ev)
    assert lookup["never_used"] and not lookup["used"] and lookup["requests"] == 2
    strict = usage.item_signals("env/x:strict", ev)
    assert not strict["used"] and not strict["never_used"]  # referenced, never called
    unseen = usage.item_signals("env/y:nothing", ev)
    assert not unseen["never_used"] and unseen["requests"] == 0


# --- Sol's first message ---------------------------------------------------------------------------------


class _First:
    """A model that answers in text only and keeps the first messages it was sent."""

    def __init__(self) -> None:
        self.first: list[dict] | None = None

    async def __call__(self, messages, tools):
        if self.first is None:
            self.first = [dict(m) for m in messages]
        return {"role": "assistant", "content": "thinking"}, "0.001"


def _first_message(tmp_path, mem, ev, eids, pass_id) -> str:
    model = _First()
    sol = SolPass(
        mem,
        Gate(mem, ev, BlobStore(tmp_path / "b")),
        ev,
        load=lambda eid: _ep(episode_id=eid),
        model_turn=model,
        config=PassConfig(max_calls=1),
    )
    asyncio.run(sol.run(PassRequest("batched", None, list(eids), False), pass_id))
    return model.first[1]["content"]


def test_sols_first_message_carries_the_usage_table_and_no_checker_text(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    base = mem.head()
    with mem.temp_checkout() as wt:
        (wt / "env/x").mkdir(parents=True)
        (wt / "env/x/__init__.py").write_text(MODULE)
        mem.fast_forward("main", mem.commit_all(wt, "seed", {}), expected_old=base)
    ev = EvidenceStore(tmp_path / "e.sqlite")
    _index(ev, "e0", _record({"env/x:parse": {"called": 1}}), "2026-10-08T00:00:00Z")
    _index(
        ev,
        "e1",
        _record(
            {
                "env/x:parse": {
                    "imported": 1,
                    "called": 2,
                    "refused": 1,
                    "refused_then_accepted": 1,
                },
            },
        ),
        "2026-10-08T01:00:00Z",
    )
    before = _first_message(tmp_path, mem, ev, ["e1"], "p1")
    ev.add_signal(
        Signal(
            "e1.checker",
            "e1",
            "checker",
            "fail",
            "2026-10-08T01:05:00Z",
            refers_to="e1",
        ),
    )
    after = _first_message(tmp_path, mem, ev, ["e1"], "p2")
    table = before.split("Current index:", 1)[1]
    assert usage.USAGE_HEADING in table
    assert "env/x:parse | 1 | 1 | 2 | 1 | 1 | 0 | 0 | 0 requests ago" in table
    assert "env/x:lookup | 1 | 0 | 0 | 0 | 0 | 0 | 0 | never" in table
    # ruling R10: the checker's verdict changes nothing Sol is shown
    assert after.split("Current index:", 1)[1] == table
    for word in ("checker", "fail", "solved", "e1.checker"):
        assert word not in table
