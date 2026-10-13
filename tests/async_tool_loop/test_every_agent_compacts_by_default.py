"""Every agent compacts by default under the one shared point: the agent loop (the actor, and every teammate started
on the agent record, which runs the same ``CodeActActor.act`` loop) and the memory agents (writer, curator and
helpers; tests/memory/test_writer_compaction.py). No setting is needed: the loop's ``enable_compression`` defaults
to on, and only the loops listed in EXEMPT turn it off."""

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

#: The loops that never compact, by their loop_id, and why: the compressor's own loop (compacting it would recurse)
#: and the storage review forked from a finished session (a fork of a conversation the session already kept under the
#: compaction point, sent as one cached prefix plus the review message).
EXEMPT = {"compress_messages", "StorageCheck(CodeActActor.act)"}


def look() -> str:
    """Look again."""
    return "looked"


def _actor(prompt_tokens: int):
    def answer(call):
        if any(SUMMARY in str(m.get("content")) for m in call.messages):
            return reply("done")
        if any(m["role"] == "tool" for m in call.messages):
            return (
                reply(calls=[("compress_context", {})])
                if prompt_tokens >= cc.COMPACTION_AT_TOKENS
                else reply("done")
            )
        return reply(calls=[("look", {})], prompt_tokens=prompt_tokens)

    return answer


async def _run(prompt_tokens: int) -> ScriptedModel:
    from unify.common.async_tool_loop import start_async_tool_loop

    answer = _actor(prompt_tokens)
    model = ScriptedModel(
        actor=Always(answer),
        compress_turn=Always(answer),
        compression_fork=[reply(SUMMARY)],
    )
    client = h.new_client(SYSTEM)
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            "Look, then answer.",
            {"look": look},
            log_steps=False,
            timeout=WAIT,
        )
        assert await asyncio.wait_for(handle.result(), WAIT) == "done"
    return model


@pytest.mark.asyncio
async def test_the_agent_loop_compacts_at_the_shared_point_with_default_settings():
    at = await _run(cc.COMPACTION_AT_TOKENS)
    assert "compression_fork" in at.kinds()
    below = await _run(cc.COMPACTION_AT_TOKENS - 1)
    assert "compression_fork" not in below.kinds() and "compactor" not in below.kinds()


def _loop_calls() -> list[tuple[str, int, str | None, bool]]:
    """Every ``start_async_tool_loop(...)`` call under unify/ (legacy code aside): (file, line, loop_id or None,
    whether it passes ``enable_compression=False``)."""
    out = []
    for path in sorted((ROOT / "unify").rglob("*.py")):
        if "legacy" in path.parts:
            continue
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
            off = (
                isinstance(kw.get("enable_compression"), ast.Constant)
                and kw["enable_compression"].value is False
            )
            out.append((str(path.relative_to(ROOT)), node.lineno, lid, off))
    return out


def test_only_the_exempt_loops_turn_compaction_off():
    import inspect

    from unify.common.async_tool_loop import start_async_tool_loop

    assert (
        inspect.signature(start_async_tool_loop)
        .parameters["enable_compression"]
        .default
        is True
    )
    calls = _loop_calls()
    assert any(lid == "CodeActActor.act" and not off for _, _, lid, off in calls), calls
    off = {lid for _, _, lid, o in calls if o}
    assert off == EXEMPT, calls


def test_a_teammate_runs_the_actor_loop():
    """A teammate started on the agent record runs ``actor.act``: the same loop, so the same default."""
    src = (ROOT / "unify" / "agents" / "binding.py").read_text(encoding="utf-8")
    assert "await actor.act(" in src
