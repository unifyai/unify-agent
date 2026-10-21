"""Every agent compacts under the one shared point: the agent loop (the actor, and every teammate started on the
agent record, which runs the same ``CodeActActor.act`` loop) and the memory agents (writer, curator and helpers;
tests/memory/test_writer_compaction.py). A loop started with ``tools_in_code`` compacts before its next model call once
a prompt reaches the point; only the loops listed in EXEMPT are started without it."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from tests import cache_discipline_helpers as h
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.common._async_tool import context_compression as cc

SYSTEM = "You are a scripted test agent."
SUMMARY = "Summary: looked once; the answer is 42."
WAIT = 30
ROOT = Path(__file__).resolve().parents[2]

#: The loops that never compact, by their loop_id, and why: the compressor's own loop (compacting it would recurse).
EXEMPT = {"compress_messages"}


def execute_code(code: str) -> str:
    """Run code (the actor's kind of loop has no other tool)."""
    return "looked"


def _actor(prompt_tokens: int):
    def answer(call):
        # after the look (or the summary of it), the answer
        if any(m["role"] == "tool" or SUMMARY in str(m.get("content")) for m in call.messages):
            return reply("done")
        return reply(calls=[("execute_code", {"code": "look()"})], prompt_tokens=prompt_tokens)

    return answer


async def _run(prompt_tokens: int) -> ScriptedModel:
    from unify.common.async_tool_loop import start_async_tool_loop

    answer = _actor(prompt_tokens)
    model = ScriptedModel(
        actor=Always(answer),
        compression_fork=[reply(SUMMARY)],
    )
    client = h.new_client(SYSTEM)
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            "Look, then answer.",
            {"execute_code": execute_code},
            log_steps=False,
            timeout=WAIT,
            tools_in_code=True,
        )
        assert await asyncio.wait_for(handle.result(), WAIT) == "done"
    return model


@pytest.mark.asyncio
async def test_the_agent_loop_compacts_at_the_shared_point():
    at = await _run(cc.COMPACTION_AT_TOKENS)
    assert "compression_fork" in at.kinds()
    below = await _run(cc.COMPACTION_AT_TOKENS - 1)
    assert "compression_fork" not in below.kinds() and "compactor" not in below.kinds()


def _loop_calls() -> list[tuple[str, int, str | None, bool]]:
    """Every ``start_async_tool_loop(...)`` call under unify/: (file, line, loop_id or None, whether it is started
    without ``tools_in_code``)."""
    out = []
    for path in sorted((ROOT / "unify").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None)
            )
            if name != "start_async_tool_loop":
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            loop_id = kw.get("loop_id")
            lid = loop_id.value if isinstance(loop_id, ast.Constant) else None
            if isinstance(loop_id, ast.JoinedStr):
                lid = "".join(
                    v.value for v in loop_id.values if isinstance(v, ast.Constant)
                )
            off = "tools_in_code" not in kw
            out.append((str(path.relative_to(ROOT)), node.lineno, lid, off))
    return out


def test_only_the_exempt_loops_are_started_without_compaction():
    calls = _loop_calls()
    on = {(f, lid) for f, _, lid, o in calls if not o}
    assert ("unify/actor/code_act_actor.py", "CodeActActor.act") in on, calls
    off = {lid for _, _, lid, o in calls if o}
    assert off == EXEMPT, calls


def test_a_teammate_runs_the_actor_loop():
    """A teammate started on the agent record runs ``actor.act``: the same loop, so the same default."""
    src = (ROOT / "unify" / "agents" / "binding.py").read_text(encoding="utf-8")
    assert "await actor.act(" in src
