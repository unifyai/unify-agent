"""One scripted office visit end to end through the CLI, on the merged online build.

The work-tree-shaped (office) variant: a workspace holds ``claims.csv``; the actor's one cell reads it and
writes ``summary.json`` (and drops a stray file into the memory export), then the actor replies. The CLI runs
in process exactly as the OFF-equivalence test drives it (``Act`` with ``act --persist --jsonl --no-clarify
--quiet``, a pipe for stdin, the scripted transport under unillm), with ``UNIFY_MEMORY=on`` and E forced
to 1 so the first episode makes a pass due; the other switches are at their defaults. After the response an
outcome whose check reason is ``SENTINEL-7f3a`` is posted, then ``{"quit": true}``.

The writer is a fake model turn put over the turn the consolidation driver builds (``consolidate.unillm_turn``): it
records every message it is sent and finishes at once with no manifest. What was staged for the writer is copied
out as it is staged (``WriterPass._stage_inputs``), so the sentinel check covers the writer's inputs too.

Checked: the jsonl lines (the accepted outcome, a consolidation start and end with ``no_manifest``, decimal
USD and the writer's effort equal to the actor's, then ``ended``); the system prompt ends with the memory section
(the v2 index under ``UNIFY_MEMORY_SURFACING=index``, the default; the constant guide under
``catalogue``, on both visits);
no review call; one episode commit whose ``actions.jsonl`` has the work-tree rows (a ``read`` of
``claims.csv`` with a csv shape and a ``write`` of ``summary.json`` on ``worktree:workspace``), whose
``cells.jsonl`` has the cell and whose meta has both snapshots; one pass/fail checker note; one ``passes``
row; both events in the state dir's ``events.jsonl``; the sentinel nowhere under the home, in the writer's messages
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
from pathlib import Path
from typing import Any


from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.scripted_model import ScriptedModel, scripted
from unify.memory import writer_pass
from unify.memory.integration import consolidate

SENTINEL = "SENTINEL-7f3a"
EFFORT = "medium"  # the actor's reasoning effort for the run
WRITER_EFFORT = EFFORT  # the writer's matches the actor's by default
WRITER_USD = "0.0000005"  # what the fake writer turn reports per call
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


class _FakeWriter:
    """The writer's model turn: records what it is sent and what was staged, finishes with no manifest."""

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
                "id": f"writer_finish_{len(self.sent)}",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps({"summary": "nothing to keep"}),
                },
            }
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            }, WRITER_USD

        return turn


def _install_fake_writer(monkeypatch) -> _FakeWriter:
    fake = _FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake.factory)
    original = writer_pass.WriterPass._stage_inputs

    def staging(self, req, inputs, *tree):
        original(self, req, inputs, *tree)
        for path in sorted(Path(inputs).rglob("*")):
            if path.is_file() and not path.is_symlink():
                fake.staged.append(path.read_bytes())

    monkeypatch.setattr(writer_pass.WriterPass, "_stage_inputs", staging)
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


