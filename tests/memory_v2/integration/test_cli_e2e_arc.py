"""Scripted Continual-ARC visits end to end through the CLI with dialogue capture (keyless).

Each visit is one ``unify act --persist --jsonl --no-clarify --quiet <first message>`` session, run in
process as the office end-to-end test runs it, with ``UNIFY_MEMORY_V2=on`` and
``UNIFY_MEMORY_V2_DIALOGUE=env``. The host answers every response line with the next runner message
(``{"message": ...}``: the benchmark's rendered feedback and observation, :mod:`tests.memory_v2.arc_transcript`)
and ends the session with ``{"quit": true}`` after the agent's ``finish``, as the baselines' runner does.
The scripted actor replies with the action JSON on its reply's last line (one turn runs a cell first).

1. Visit 1 (E high): its episode holds the dialogue actions on ``env`` (request_demos, submit, submit,
   finish), each answered by the runner message that followed, and E counts each observation once. No pass.
2. E is set just above visit 1's experience, so visit 2's episode makes a batched pass due over both. Sol is
   a scripted model turn (``consolidate.unillm_turn``) whose one cell writes ``env/env:observation_lines``
   covering every recorded observation, then finishes. The pass passes the gate and merges on ``env/env``.
3. Visit 3's export holds the merged item and its system prompt's index lists it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sqlite3
import sys
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
from tests.memory_v2.arc_transcript import (
    ITEM,
    SOL_CELL,
    demos_feedback,
    grid,
    observation,
    submit_feedback,
)
from tests.memory_v2.test_sol_pass import Script
from tests.scripted_model import ScriptedModel, _text, cell, reply, scripted
from unify import sandbox
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import episode_dir, load_episode
from unify.memory_v2.experience import experience_tokens
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import HEADER
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
from unify.settings import SETTINGS

EFFORT = "medium"
SOL_USD = "0.0000005"
HIGH_E = 10**9


def _act(thought: str, action: dict) -> str:
    return f"{thought}\n{json.dumps(action)}"


def _visit_messages(task: str, seed: int, *, solve_first: bool) -> dict:
    """The first message, the runner's later messages, and the actor's replies for one visit."""
    test_input = grid(3, 3, seed=seed)
    pair = (grid(3, 3, seed=seed + 1), grid(3, 3, seed=seed + 2))
    wrong, right = grid(3, 3, seed=seed + 3), grid(3, 3, seed=seed + 4)
    first = observation(task, test_input, first=True)
    demos = observation(task, test_input, demos=1, pending=(demos_feedback([pair], 1),))
    if solve_first:
        messages = [
            demos,
            observation(task, test_input, demos=1, pending=(submit_feedback(True, 0),)),
        ]
        actor = [
            reply(_act("I ask for a demonstration.", {"action": "request_demos"})),
            reply(_act("The rule is clear.", {"action": "submit", "grid": right})),
            reply(json.dumps({"action": "finish"})),
        ]
        methods = ["request_demos", "submit", "finish"]
    else:
        messages = [
            demos,
            observation(
                task,
                test_input,
                attempts=1,
                demos=1,
                pending=(submit_feedback(False, 1),),
            ),
            observation(
                task,
                test_input,
                attempts=1,
                demos=1,
                pending=(submit_feedback(True, 1),),
            ),
        ]
        actor = [
            reply(_act("I ask for a demonstration.", {"action": "request_demos"})),
            reply(calls=[cell("print('rows', 3)")]),
            reply(_act("Each cell shifts.", {"action": "submit", "grid": wrong})),
            reply(_act("Second try.", {"action": "submit", "grid": right})),
            reply(json.dumps({"action": "finish"})),
        ]
        methods = ["request_demos", "submit", "submit", "finish"]
    return {"first": first, "messages": messages, "actor": actor, "methods": methods}


async def _arc_visit(
    world,
    monkeypatch,
    model: ScriptedModel,
    first: str,
    messages: list[str],
):  # noqa: F811
    """One CLI session: each response line is answered by the next runner message, then quit."""
    from unify.cli import Act, _parse_args

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    args = _parse_args(
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", first],
    )
    session = Act(args)
    lines: list[dict] = []
    queue = list(messages)
    closed = False

    def emit(**payload: Any) -> None:
        nonlocal closed
        lines.append(json.loads(json.dumps(payload, default=str)))
        if payload.get("type") != "response" or closed:
            return
        if queue:
            os.write(write_fd, (json.dumps({"message": queue.pop(0)}) + "\n").encode())
        else:
            closed = True
            os.write(write_fd, b'{"quit": true}\n')
            os.close(write_fd)

    async def start() -> None:
        os.chdir(world["workspace"])
        session._actor = new_actor()

    monkeypatch.setattr(session, "_emit", emit)
    monkeypatch.setattr(session, "start", start)
    try:
        with scripted(model):
            code = await asyncio.wait_for(session.run(first), 300)
    finally:
        await session.close()
        if not closed:
            os.close(write_fd)
    return code, lines


def _query(db: Path, sql: str, *params) -> list[tuple]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def _load(paths: Paths, eid: str):
    ((sha, started_at),) = _query(
        paths.evidence,
        "SELECT commit_sha, started_at FROM episodes WHERE episode_id = ?",
        eid,
    )
    rel = episode_dir(SimpleNamespace(episode_id=eid, started_at=started_at))
    return load_episode(Repo(paths.episodes), sha, rel, BlobStore(paths.blobs))


def _episode_ids(paths: Paths) -> list[str]:
    return [
        r[0]
        for r in _query(paths.evidence, "SELECT episode_id FROM episodes ORDER BY seq")
    ]


def _types(lines: list[dict]) -> list[str]:
    """The line types, without storage lines or record posts to @user (none is scripted here)."""
    return [line["type"] for line in lines if line["type"] not in ("storage", "record")]


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file()
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@_handle_project
async def test_arc_visits_capture_dialogue_and_merge_an_env_item(
    core_world,
    monkeypatch,
):
    from unify.session_details import SESSION_DETAILS

    home = core_world["state"]
    paths = Paths.under(home)
    defaults = type(SETTINGS).model_fields
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_DIALOGUE", "env")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_E", HIGH_E)
    for name in (
        "UNIFY_MEMORY_V2_SOL_MODEL",
        "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS",
        "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD",
        "UNIFY_MEMORY_V2_SOL_EFFORT",
        "UNIFY_MEMORY_V2_SOL_EFFORT_SCALE",
        "UNIFY_MEMORY_V2_SOL_MAX_CALLS",
    ):
        monkeypatch.setattr(SETTINGS, name, defaults[name].default)
    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", EFFORT)
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    script = Script([SOL_CELL], usd=SOL_USD)
    sol_calls: list[tuple[str, str]] = []

    def sol_turn(model: str, effort: str, **_kw: Any):
        sol_calls.append((model, effort))
        return script

    monkeypatch.setattr(consolidate, "unillm_turn", sol_turn)

    # ── visit 1: dialogue actions recorded, no pass ─────────────────────────
    v1 = _visit_messages("task-3f2a9c1e", 0, solve_first=False)
    model1 = ScriptedModel(actor=v1["actor"])
    code, lines = await _arc_visit(
        core_world,
        monkeypatch,
        model1,
        v1["first"],
        v1["messages"],
    )
    assert code == 0, lines
    types = _types(lines)
    assert types == ["response"] * 4 + ["result", "ended"], lines
    assert model1.kinds() == ["actor"] * 5
    (eid1,) = _episode_ids(paths)
    ep1 = _load(paths, eid1)
    acts1 = [a for a in ep1.actions if a.kind == "dialogue"]
    assert [a.method for a in acts1] == v1["methods"]
    assert {(a.channel, a.cell) for a in acts1} == {("env", -1)}
    assert [a.status for a in acts1[:3]] == ["ok"] * 3
    assert acts1[0].args == [] and acts1[3].args == []
    assert acts1[1].args == [grid(3, 3, seed=3)] and acts1[2].args == [
        grid(3, 3, seed=4),
    ]
    # each action's observation is the runner message that answered it, as the actor received it
    for act, message in zip(acts1[:3], v1["messages"]):
        assert isinstance(act.response, str) and message in act.response
    # E counts each observation once: the dialogue actions add nothing to the request messages
    bare = dataclasses.replace(
        ep1,
        actions=[a for a in ep1.actions if a.kind != "dialogue"],
    )
    assert experience_tokens(ep1) == experience_tokens(bare)
    ((t1,),) = _query(
        paths.evidence,
        "SELECT tokens FROM experience WHERE episode_id = ?",
        eid1,
    )
    assert "env.request_demos" in ep1.fingerprints
    assert not paths.events.exists() and sol_calls == []

    # ── visit 2: a batched pass over both episodes merges an env/env item ───
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_E", t1 + 1)
    v2 = _visit_messages("task-0badf00d", 5, solve_first=True)
    model2 = ScriptedModel(actor=v2["actor"])
    code, lines = await _arc_visit(
        core_world,
        monkeypatch,
        model2,
        v2["first"],
        v2["messages"],
    )
    assert code == 0, lines
    types = _types(lines)
    assert types == ["response"] * 3 + [
        "result",
        "consolidation",
        "consolidation",
        "ended",
    ], lines
    eid1_again, eid2 = _episode_ids(paths)
    assert eid1_again == eid1
    start, end = (line for line in lines if line["type"] == "consolidation")
    assert start["phase"] == "start" and end["phase"] == "end"
    assert start["episodes"] == [eid1, eid2] and start["pass_id"] == end["pass_id"]
    assert start["trigger_tokens"] >= t1 + 1
    assert sol_calls == [(defaults["UNIFY_MEMORY_V2_SOL_MODEL"].default, EFFORT)]
    assert "EPISODES 2" in script.outputs["c1"], script.outputs
    assert end["gate_passed"] is True, end
    assert end["reason_codes"] == ["ok"] and end["items"] == 1, end
    memory = Repo(paths.memory)
    tree = memory.run("ls-tree", "-r", "--name-only", "main").split()
    assert "env/env/__init__.py" in tree
    assert ITEM.split(":")[1] in memory.show("main", "env/env/__init__.py").decode()
    ep2 = _load(paths, eid2)
    want = {
        (ep.episode_id, i)
        for ep in (ep1, ep2)
        for i, a in enumerate(ep.actions)
        if a.kind == "dialogue" and a.status == "ok" and isinstance(a.response, str)
    }
    covers = {
        (r[0], int(r[1]))
        for r in _query(paths.evidence, "SELECT episode_id, action_index FROM covers")
    }
    assert len(want) >= 5 and covers == want
    passes = _query(paths.evidence, "SELECT pass_id, passed FROM passes")
    assert passes == [(start["pass_id"], 1)]

    # ── visit 3: the merged item is exported and indexed ────────────────────
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_E", HIGH_E)
    exports: list[list[str]] = []

    def first_turn(call):
        exports.append(_files(paths.checkout))
        return reply(_act("I ask for a demonstration.", {"action": "request_demos"}))

    model3 = ScriptedModel(actor=[first_turn])
    v3 = _visit_messages("task-3f2a9c1e", 9, solve_first=True)
    code, lines = await _arc_visit(core_world, monkeypatch, model3, v3["first"], [])
    assert code == 0, lines
    assert exports and "env/env/__init__.py" in exports[0], exports
    system = "\n".join(
        _text(m.get("content"))
        for m in model3.of("actor")[0].messages
        if m.get("role") == "system"
    )
    assert HEADER in system
    assert "observation_lines" in system[system.rindex(HEADER) :]
