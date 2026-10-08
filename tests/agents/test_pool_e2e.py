"""Symbolic: a main agent and helpers in worker sandboxes, all through the record.

Scripted models only. Each agent's requests are routed to its own script by
who is asking: a helper's conversation opens with "You are `<name>`".
"""

import asyncio
import contextlib
import json
import os

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.environments import actor as actor_env
from unify.settings import SETTINGS

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(300)]


def _who(messages) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        text = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        for name in ("h1", "h2"):
            if f"You are `{name}`" in text:
                return name
    return "root"


class Routed(h.Provider):
    """Plays each agent's own script, chosen by who is asking."""

    def __init__(self, scripts):
        super().__init__()
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.by_agent = {k: [] for k in scripts}

    async def __call__(self, *, shared_session=None, client=None, **kw):
        who = _who(kw["messages"])
        self.by_agent[who].append(kw)
        self.requests.append(kw)
        if not self.scripts[who]:
            raise AssertionError(f"no scripted reply left for {who}")
        reply = self.scripts[who].pop(0)
        return reply() if callable(reply) else reply


@contextlib.contextmanager
def _routed(scripts):
    """Install ``Routed`` as unillm's transport, as ``h.scripted`` does."""
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    provider = Routed(scripts)
    original = uni_llm._acompletion_with_transient_retry
    old_cache = os.environ.get("UNILLM_CACHE")
    old_default = unillm_settings.UNILLM_CACHE
    uni_llm._acompletion_with_transient_retry = provider
    os.environ["UNILLM_CACHE"] = "false"
    unillm_settings.UNILLM_CACHE = False
    try:
        yield provider
    finally:
        uni_llm._acompletion_with_transient_retry = original
        unillm_settings.UNILLM_CACHE = old_default
        if old_cache is None:
            os.environ.pop("UNILLM_CACHE", None)
        else:
            os.environ["UNILLM_CACHE"] = old_cache


def _code(src):
    return lambda: h.completion(calls=[("execute_code", {"thought": "t", "code": src})])


def _text(content):
    return lambda: h.completion(content=content)


async def _run(scripts, monkeypatch, tmp_path):
    from unify.actor.code_act_actor import CodeActActor
    from unify.agents import binding

    # As record mode will be screened: on lean-all, whose discovery gate is off
    # (the shipped gate forces a library search before any reply).
    for key, value in {
        "UNIFY_AGENTS": "record",
        "UNIFY_WORKSPACE": "sandboxed",
        "UNIFY_WORKSPACE_PYTHON": "worker",
    }.items():
        monkeypatch.setattr(SETTINGS, key, value)
    monkeypatch.setattr(binding, "records_dir", lambda: tmp_path / "records")
    actor = CodeActActor(
        environments=actor_env.top_level_environments(),
        tool_policy=None,
    )
    pool = None
    try:
        with _routed(scripts) as provider:
            handle = await actor.act(
                "Summarise a.csv and b.csv.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            pool = handle.agents_pool
            result = await asyncio.wait_for(handle.result(), 240)
    finally:
        if pool is not None:
            await pool.close("test over")
        await actor.close()
    return provider, result, pool


async def test_fan_out_wait_and_reply(monkeypatch, tmp_path):
    root = [
        _code(
            "a = await agents.spawn('Summarise a.csv: rows and columns please.')\n"
            "b = await agents.spawn('Summarise b.csv: rows and columns please.')\n"
            "got = {}\n"
            "while len(got) < 2:\n"
            "    for e in await record.wait(timeout=120):\n"
            "        if e['kind'] == 'reply':\n"
            "            got[e['author']] = e['text']\n"
            "print(sorted(got.items()))",
        ),
        _text("a: 3 rows; b: 5 rows"),
    ]
    scripts = {"root": root, "h1": [_text("3 rows")], "h2": [_text("5 rows")]}
    provider, result, pool = await _run(scripts, monkeypatch, tmp_path)
    assert result == "a: 3 rows; b: 5 rows"
    seen = [(e.seq, e.author, e.kind, e.text[:120]) for e in pool.record.entries]
    replies = [(e.author, e.mentions) for e in pool.record.entries if e.kind == "reply"]
    assert ("h1", ("root",)) in replies and ("h2", ("root",)) in replies, seen
    assert ("root", ("user",)) in replies
    assert len(provider.by_agent["root"]) == 2  # waiting cost no model call
    assert "primitives.actor" not in json.dumps(provider.by_agent["h1"][0], default=str)


async def test_a_helper_cannot_spawn_or_post_as_someone_else(monkeypatch, tmp_path):
    root = [
        _code(
            "await agents.spawn('Try to spawn and to post as another agent, please.')\n"
            "print(await record.wait(timeout=120))",
        ),
        _text("done"),
    ]
    helper = [
        _code(
            "try:\n"
            "    await agents.spawn('another helper for this request now')\n"
            "except Exception as e:\n"
            "    print(type(e).__name__, e)\n"
            "try:\n"
            "    record._name = 'root'\n"
            "except Exception as e:\n"
            "    print('refused', type(e).__name__)\n"
            "record.post('@root hello')",
        ),
        _text("tried"),
    ]
    provider, result, pool = await _run(
        {"root": root, "h1": helper},
        monkeypatch,
        tmp_path,
    )
    helper_seen = json.dumps(provider.by_agent["h1"][-1]["messages"], default=str)
    assert "only the main agent can start helpers" in helper_seen
    hello = [e for e in pool.record.entries if e.text == "@root hello"]
    assert hello and hello[0].author == "h1"


async def test_agents_waiting_on_each_other_are_released_by_their_timeouts(
    monkeypatch,
    tmp_path,
):
    root = [
        _code(
            "await agents.spawn('Wait for the main agent to say go, then reply.')\n"
            "print(await record.wait(timeout=2))",
        ),
        _text("gave up waiting"),
    ]
    helper = [_code("print(await record.wait(timeout=2))"), _text("no go")]
    provider, result, pool = await _run(
        {"root": root, "h1": helper},
        monkeypatch,
        tmp_path,
    )
    assert result == "gave up waiting"
    assert all(task.done() for task in pool._tasks.values())
