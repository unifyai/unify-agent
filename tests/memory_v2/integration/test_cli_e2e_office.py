"""One scripted office visit end to end through the CLI, on the merged online build (integration Task 27).

The work-tree-shaped (office) variant: a workspace holds ``claims.csv``; the actor's one cell reads it and
writes ``summary.json`` (and drops a stray file into the memory export), then the actor replies. The CLI runs
in process exactly as the OFF-equivalence test drives it (``Act`` with ``act --persist --jsonl --no-clarify
--quiet``, a pipe for stdin, the scripted transport under unillm), with ``UNIFY_MEMORY_V2=on`` and E forced
to 1 so the first episode makes a pass due; the other switches are at their defaults. After the response an
outcome whose check reason is ``SENTINEL-7f3a`` is posted, then ``{"quit": true}``.

Sol is a fake model turn put over the turn the consolidation driver builds (``consolidate.unillm_turn``): it
records every message it is sent and finishes at once with no manifest. What was staged for Sol is copied
out as it is staged (``SolPass._stage_inputs``), so the sentinel check covers Sol's inputs too.

Checked: the jsonl lines (the accepted outcome, a consolidation start and end with ``no_manifest``, decimal
USD and Sol's effort equal to the actor's, then ``ended``); the system prompt ends with the memory index;
no review call; one episode commit whose ``actions.jsonl`` has the work-tree rows (a ``read`` of
``claims.csv`` with a csv shape and a ``write`` of ``summary.json`` on ``worktree:workspace``), whose
``cells.jsonl`` has the cell and whose meta has both snapshots; one pass/fail checker note; one ``passes``
row; both events in the state dir's ``events.jsonl``; the sentinel nowhere under the home, in Sol's messages
or in its staged inputs; the export removed after the request, and a second visit's export as clean as the
first one was before its cell ran.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from tests.memory_v2.integration.fake_tracks import dump_home
from tests.memory_v2.integration.test_checkout import _seed
from tests.scripted_model import ScriptedModel, _text, cell, reply, scripted
from unify import sandbox
from unify.memory_v2 import sol_pass
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import episode_dir, load_episode
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import HEADER
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.prompt import export_line
from unify.settings import SETTINGS

SENTINEL = "SENTINEL-7f3a"
EFFORT = "medium"  # the actor's reasoning effort for the run
SOL_EFFORT = "low"  # Sol's: the memory system's declared constant, never the actor's (D23 as revised 8 Oct)
SOL_USD = "0.0000005"  # what the fake Sol turn reports per call
CHANNEL = "worktree:workspace"
STRAY = "env/stray_note.py"
PLAIN_DECIMAL = re.compile(r"^[0-9]+(\.[0-9]+)?\Z")

REQUEST = "Summarise claims.csv into summary.json."
CLAIMS = "id,claimant,amount\n1,ada,120.50\n2,bob,80\n3,cy,15.25\n"
FINAL = "Wrote summary.json: 3 claims."
OUTCOME = {
    "solved": False,
    "checks": [{"name": "c", "passed": False, "reason": SENTINEL}],
}


def _code(export: Path) -> str:
    return (
        "import csv, json\n"
        "with open('claims.csv') as fh:\n"
        "    rows = list(csv.DictReader(fh))\n"
        "with open('summary.json', 'w') as fh:\n"
        "    json.dump({'claims': len(rows), 'total': sum(float(r['amount']) for r in rows)}, fh)\n"
        f"with open({str(export / STRAY)!r}, 'w') as fh:\n"
        "    fh.write('x = 1\\n')\n"
        "print(len(rows), 'claims')\n"
    )


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()
    )


class _FakeSol:
    """Sol's model turn: records what it is sent and what was staged, finishes with no manifest."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.efforts: list[str] = []
        self.sent: list[str] = []
        self.staged: list[bytes] = []

    def factory(self, model: str, effort: str, **_kw: Any):
        self.models.append(model)
        self.efforts.append(effort)

        async def turn(messages: list[dict], tools: list[dict]) -> tuple[dict, str]:
            self.sent.append(json.dumps(messages, default=str))
            call = {
                "id": f"sol_finish_{len(self.sent)}",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps({"summary": "nothing to keep"}),
                },
            }
            return {"role": "assistant", "content": "", "tool_calls": [call]}, SOL_USD

        return turn


def _install_fake_sol(monkeypatch) -> _FakeSol:
    fake = _FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake.factory)
    original = sol_pass.SolPass._stage_inputs

    def staging(self, req, inputs):
        original(self, req, inputs)
        for path in sorted(Path(inputs).rglob("*")):
            if path.is_file() and not path.is_symlink():
                fake.staged.append(path.read_bytes())

    monkeypatch.setattr(sol_pass.SolPass, "_stage_inputs", staging)
    return fake


async def _visit(
    world,
    monkeypatch,
    model: ScriptedModel,
    request: str,
    *,
    outcome=None,
):  # noqa: F811
    """One CLI visit (``Act`` in process); returns (exit code, the jsonl lines parsed)."""
    from unify.cli import Act, _parse_args

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    args = _parse_args(
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", request],
    )
    session = Act(args)
    lines: list[str] = []
    closed = False

    def emit(**payload: Any) -> None:
        nonlocal closed
        lines.append(json.dumps(payload, default=str))
        if payload.get("type") == "response" and not closed:
            closed = True
            if outcome is not None:
                os.write(write_fd, (json.dumps({"outcome": outcome}) + "\n").encode())
            os.write(write_fd, b'{"quit": true}\n')
            os.close(write_fd)

    async def start() -> None:
        os.chdir(world["workspace"])
        session._actor = new_actor()

    monkeypatch.setattr(session, "_emit", emit)
    monkeypatch.setattr(session, "start", start)
    try:
        with scripted(model):
            code = await asyncio.wait_for(session.run(request), 120)
    finally:
        await session.close()
        if not closed:
            os.close(write_fd)
    return code, [json.loads(x) for x in lines]


def _query(db: Path, sql: str) -> list[tuple]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_one_office_visit_end_to_end(core_world, monkeypatch):
    from unify.session_details import SESSION_DETAILS

    home = core_world["state"]
    paths = Paths.under(home)
    (core_world["workspace"] / "claims.csv").write_text(CLAIMS)
    _seed(home)

    defaults = type(SETTINGS).model_fields
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_MEMORY_V2_E",
        1,
    )  # the first episode makes a pass due
    for name in (
        "UNIFY_MEMORY_V2_SOL_MODEL",
        "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS",
        "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD",
    ):
        monkeypatch.setattr(SETTINGS, name, defaults[name].default)
    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", EFFORT)
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    fake = _install_fake_sol(monkeypatch)

    # ── visit 1 ──────────────────────────────────────────────────────────────
    exports: list[list[str]] = (
        []
    )  # the export's files when each visit's actor first runs

    def first_turn(call):
        exports.append(_files(paths.checkout))
        return reply(calls=[cell(_code(paths.checkout))])

    model = ScriptedModel(actor=[first_turn, reply(FINAL)])
    code, lines = await _visit(core_world, monkeypatch, model, REQUEST, outcome=OUTCOME)

    # 1. exit 0; the accepted outcome, a consolidation start and end, then ended
    assert code == 0, lines
    types = [line["type"] for line in lines]
    assert types == [
        "response",
        "outcome",
        "result",
        "consolidation",
        "consolidation",
        "ended",
    ], lines
    outcome_line = lines[types.index("outcome")]
    assert outcome_line == {
        "type": "outcome",
        "accepted": True,
        "solved": False,
        "checks": 1,
    }
    start, end = (line for line in lines if line["type"] == "consolidation")
    assert start["phase"] == "start" and end["phase"] == "end"
    assert start["pass_id"] == end["pass_id"]
    eid = start["episodes"][0]
    assert start["episodes"] == [eid] and start["pass_id"] == f"{eid}.p0"
    assert start["sol_model"] == defaults["UNIFY_MEMORY_V2_SOL_MODEL"].default
    assert (
        start["cap_usd"] == "0.00000073"
    )  # E x the default allowance, a plain decimal
    assert start["trigger_tokens"] >= 1
    assert "no_manifest" in end["reason_codes"], end
    assert end["gate_passed"] is False and end["calls"] == 1
    assert isinstance(end["usd"], str) and PLAIN_DECIMAL.match(end["usd"]), end
    assert Decimal(end["usd"]) == Decimal(SOL_USD)
    # Sol's effort is the declared constant, not the actor's: the event and the turn Sol was given
    assert start["sol_effort"] == SOL_EFFORT
    assert fake.efforts == [SOL_EFFORT] and len(fake.sent) == 1
    assert fake.models == [start["sol_model"]]
    actor_calls = model.of("actor")
    assert actor_calls[0].request.get("reasoning_effort") in (None, EFFORT)

    # 2. the system prompt ends with the memory index; no review (or any other) model call
    assert model.kinds() == ["actor", "actor"]
    system = "\n".join(
        _text(m.get("content"))
        for m in actor_calls[0].messages
        if m.get("role") == "system"
    )
    assert HEADER in system
    tail = system[system.rindex(HEADER) :]
    assert "hello(apis, name)" in tail
    assert tail.rstrip().endswith(export_line(paths.checkout).rstrip()), tail[-400:]

    # 3. one episode commit with the work-tree rows, the cell and both snapshots
    episodes = Repo(paths.episodes)
    assert episodes.run("log", "--format=%s", "main").splitlines() == [
        f"episode {eid}",
        "init",
    ]
    ((sha, started_at),) = _query(
        paths.evidence,
        "SELECT commit_sha, started_at FROM episodes",
    )
    rel = episode_dir(SimpleNamespace(episode_id=eid, started_at=started_at))
    ep = load_episode(episodes, sha, rel, BlobStore(paths.blobs))
    raw_actions = [
        json.loads(x)
        for x in episodes.show(sha, f"{rel}/actions.jsonl").decode().splitlines()
        if x.strip()
    ]
    rows = [a for a in raw_actions if a["kind"] == "worktree"]
    reads = [a for a in rows if a["method"] == "read" and a["args"] == ["claims.csv"]]
    writes = [
        a for a in rows if a["method"] == "write" and a["args"] == ["summary.json"]
    ]
    assert len(reads) == 1 and len(writes) == 1, raw_actions
    read, write = reads[0], writes[0]
    assert read["channel"] == write["channel"] == CHANNEL
    assert read["status"] == "ok" and read["effect"] == "read"
    assert read["response"]["shape"]["format"] == "csv"
    assert read["response"]["shape"]["delimiter"] == ","
    assert read["response"]["shape"]["columns"] == ["id", "claimant", "amount"]
    assert write["effect"] == "write"
    assert write["response"]["blob_before"] is None
    assert write["response"]["blob_after"]
    assert write["response"]["shape"]["format"] == "json"
    assert (
        read["cell"] == write["cell"] == 0
    )  # attributed to the one cell by the harness clock
    assert not any(
        STRAY in json.dumps(a) for a in raw_actions
    )  # the export is not the work tree
    assert len(ep.cells) == 1
    assert ep.cells[0].code == _code(paths.checkout)
    assert "3 claims" in ep.cells[0].output
    assert ep.replies[-1] == FINAL
    assert ep.worktree_before and ep.worktree_after
    assert ep.worktree_before != ep.worktree_after
    assert ep.effort == EFFORT
    assert (
        STRAY in ep.memory_diff
    )  # the cell's write into the export was seen, then discarded

    # 4. one pass/fail checker note, one passes row, both events in the state dir
    signals = episodes.notes(sha)
    assert len(signals) == 1, signals
    note = json.loads(signals[0])
    assert (note["source"], note["label"]) == ("checker", "fail")
    passes = _query(paths.evidence, "SELECT pass_id, passed, reasons FROM passes")
    assert len(passes) == 1 and passes[0][:2] == (start["pass_id"], 0), passes
    assert "no manifest" in passes[0][2]
    costs = [json.loads(x) for x in episodes.notes(sha, ref="costs")]
    assert [(c["purpose"], c["usd"], c["sol_effort"]) for c in costs] == [
        ("sol", SOL_USD, SOL_EFFORT),
    ]
    events = [json.loads(x) for x in paths.events.read_text().splitlines() if x.strip()]
    assert events == [start, end]

    # 6 (first half). the export is removed after the request
    assert not paths.checkout.exists()
    assert (
        exports and STRAY not in exports[0] and "env/spotify/__init__.py" in exports[0]
    )

    # ── visit 2: its export is the clean library, without the stray file ─────
    def second_turn(call):
        exports.append(_files(paths.checkout))
        return reply("Nothing to do.")

    model2 = ScriptedModel(actor=[second_turn])
    code2, lines2 = await _visit(core_world, monkeypatch, model2, "Anything new?")
    assert code2 == 0, lines2
    assert model2.kinds() == ["actor"]
    assert [line["type"] for line in lines2][-1] == "ended"
    assert len(exports) == 2 and exports[1] == exports[0], exports
    assert not paths.checkout.exists()

    # 5. the sentinel is nowhere: the home (files, every git object incl. notes, the evidence db rows),
    # Sol's messages and staged inputs, and the CLI's output
    assert fake.staged, "nothing was staged for Sol"
    assert any(b"claims.csv" in b for b in fake.staged)  # the episode did reach Sol
    dump = dump_home(home)
    assert dump  # the dump read something
    assert SENTINEL.encode() not in dump
    assert not any(SENTINEL in m for m in fake.sent)
    assert not any(SENTINEL.encode() in b for b in fake.staged)
    assert SENTINEL not in json.dumps(lines) + json.dumps(lines2)
