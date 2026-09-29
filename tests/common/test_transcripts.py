"""Symbolic: ``UNIFY_TRANSCRIPTS`` keeps an append-only transcript of every session.

The loops here are real ``start_async_tool_loop`` runs on real ``unillm``
clients whose transport is replaced by a script (as in
``test_tool_choice_fallback.py``): every request is recorded and nothing leaves
the process. Which reply a request gets depends only on which conversation sent
it, so the runs are deterministic however the event loop interleaves them.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import gc
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import unillm
from openai.types.chat import ChatCompletion

from unify import transcripts
from unify.common._async_tool import context_compression as cc
from unify.common._async_tool.event_bus_util import to_event_bus
from unify.common._async_tool.loop_config import LoopConfig
from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from unify.settings import ProductionSettings, SETTINGS

MODEL = "openai/gpt-5.6-sol@openrouter"
ROOT_PROMPT = "You are the root agent."
SUB_PROMPT = "You are the sub agent."
FAKE_TOKEN = "tok-test-9f8e7d6c5b4a3210"  # pragma: allowlist secret


def completion(content=None, calls=()) -> ChatCompletion:
    tool_calls = [
        {
            "id": f"call_{name}_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    },
                },
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


class Provider:
    """Scripted transport: one reply queue per conversation, chosen by its system prompt."""

    def __init__(self, scripts: dict[str, list[ChatCompletion]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        system = next(
            (m.get("content") for m in messages if m.get("role") == "system"),
            "",
        )
        self.requests.append(copy.deepcopy({"messages": messages}))
        for prefix, replies in self.scripts.items():
            if isinstance(system, str) and system.startswith(prefix):
                return replies.pop(0)
        raise AssertionError(f"no script for system prompt {system[:60]!r}")


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("UNILLM_CACHE", "false")
    # Clients built inside the harness (the compressor's) take the process
    # default; nothing here may reach, or be answered from, the cache.
    monkeypatch.setattr(unillm.SETTINGS, "UNILLM_CACHE", False)

    def install(scripts) -> Provider:
        p = Provider(scripts)
        monkeypatch.setattr(
            "unillm.clients.uni_llm._acompletion_with_transient_retry",
            p,
        )
        return p

    return install


def llm(system: str):
    c = new_llm_client(MODEL, stateful=True, cache=False, reasoning_effort="low")
    c.set_system_message(system)
    return c


def home() -> Path:
    return Path(os.environ["UNIFY_HOME"])


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def session_files() -> list[Path]:
    d = home() / "transcripts"
    return sorted(p for p in d.glob("*.jsonl") if p.name != "index.jsonl")


def index_lines() -> list[dict]:
    path = home() / "transcripts" / "index.jsonl"
    return read_lines(path) if path.exists() else []


def end_all_sessions():
    """Release this test's clients so each of its sessions writes its index line."""
    gc.collect()
    for session in list(transcripts._LIVE):
        if session.path.parent == home() / "transcripts":
            session.finalize()


def isolated(fn, *args):
    """Run *fn* in a context of its own, as a loop's task would."""
    return contextvars.copy_context().run(fn, *args)


async def run_delegation(provider) -> Provider:
    """A root loop whose one tool runs a sub-agent loop on its own client."""
    p = provider(
        {
            ROOT_PROMPT: [
                completion(calls=[("delegate", {"task": "count"})]),
                completion(calls=[("broken", {})]),
                completion(content="root done"),
            ],
            SUB_PROMPT: [completion(content=f"sub done with {FAKE_TOKEN}")],
        },
    )

    async def delegate(task: str) -> str:
        """Hand a task to a sub-agent and return its answer."""
        handle = start_async_tool_loop(
            llm(SUB_PROMPT),
            f"please {task}",
            {},
            loop_id="SubAgent.act",
            enable_compression=False,
        )
        return await handle.result()

    def broken() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    handle = start_async_tool_loop(
        llm(ROOT_PROMPT),
        "start",
        {"delegate": delegate, "broken": broken},
        loop_id="RootAgent.act",
        enable_compression=False,
    )
    assert await handle.result() == "root done"
    return p


# ── off: exactly as shipped ─────────────────────────────────────────────────


def test_setting_defaults_off_and_parses_on():
    assert ProductionSettings().UNIFY_TRANSCRIPTS is False
    assert ProductionSettings(UNIFY_TRANSCRIPTS="on").UNIFY_TRANSCRIPTS is True


@pytest.mark.asyncio
async def test_off_writes_nothing_and_the_model_sees_the_same_requests(
    monkeypatch,
    provider,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", False)
    off = await run_delegation(provider)
    end_all_sessions()
    assert not (home() / "transcripts").exists()

    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    on = await run_delegation(provider)
    end_all_sessions()
    assert (home() / "transcripts").exists()
    # Recording never changes what reaches the model.
    assert on.requests == off.requests


def test_off_attach_touches_nothing(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", False)

    class Client:
        messages: list = []
        system_message = "s"

    client = Client()
    cfg = LoopConfig("x", None, [])
    assert isolated(transcripts.attach, client, cfg) is None
    assert not hasattr(client, "_unify_transcript")
    assert not hasattr(cfg, "_unify_transcript")
    assert transcripts.session_for_messages(client.messages) is None
    assert not (home() / "transcripts").exists()


# ── on: one file per session, linked through the index ──────────────────────


@pytest.mark.asyncio
async def test_every_session_gets_its_own_file_and_index_line(monkeypatch, provider):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    monkeypatch.setenv("FAKE_SERVICE_TOKEN", FAKE_TOKEN)
    await run_delegation(provider)
    end_all_sessions()

    files = session_files()
    assert len(files) == 2
    by_origin = {read_lines(f)[0]["origin"]: read_lines(f) for f in files}
    root, sub = by_origin["RootAgent.act"], by_origin["SubAgent.act"]

    # Each file opens with its session and system prompt, then the messages in
    # the order the conversation had them.
    assert root[0]["type"] == "session_start" and root[0]["parent"] is None
    assert root[0]["system_prompt"] == ROOT_PROMPT
    assert sub[0]["parent"] == root[0]["session"]
    assert [ln["seq"] for ln in root] == list(range(len(root)))
    roles = [
        (ln["message"].get("role"), ln["message"].get("content"))
        for ln in root
        if ln["type"] == "message" and ln["in_context"]
    ]
    assert ("user", "start") in roles
    assert roles[-1] == ("assistant", "root done")

    # The sub-agent's answer is in both: its own final message, and the
    # delegate tool's result in the root.
    def text(lines):
        return "\n".join(json.dumps(ln) for ln in lines)

    assert "sub done with" in text(sub) and "sub done with" in text(root)
    # The credential never reaches disk; its name says what was removed.
    assert FAKE_TOKEN not in text(sub) + text(root)
    assert "[REDACTED:FAKE_SERVICE_TOKEN]" in text(sub)
    # The failing tool's traceback is kept verbatim.
    assert "RuntimeError: boom" in text(root)

    index = {line["session"]: line for line in index_lines()}
    assert len(index) == 2
    root_ix = index[root[0]["session"]]
    sub_ix = index[sub[0]["session"]]
    assert sub_ix["parent"] == root_ix["session"]
    assert root_ix["origin"] == "RootAgent.act"
    assert root_ix["tools"] == ["broken", "delegate"]
    assert root_ix["errors"] == 1 and sub_ix["errors"] == 0
    assert root_ix["started_at"] <= root_ix["ended_at"]
    assert Path(root_ix["path"]) in files


def test_lines_are_only_ever_appended(monkeypatch):
    """A placeholder filled in place becomes a new line; earlier bytes never change."""
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)

    class Client:
        system_message = "sys"

        def __init__(self):
            self.messages: list[dict] = []

    async def scenario():
        client = Client()
        cfg = LoopConfig("Loop", None, [])
        LoopMessageDispatcher(client, cfg, timer=_NoTimer())
        session = client._unify_transcript
        offsets, snapshots = [], []

        def checkpoint():
            data = session.path.read_bytes()
            offsets.append(len(data))
            snapshots.append(data)

        checkpoint()
        user = {"role": "user", "content": "hi"}
        client.messages.append(user)
        await to_event_bus(user, cfg)
        checkpoint()
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                },
            ],
        }
        placeholder = {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "t",
            "content": json.dumps({"_placeholder": "pending"}),
        }
        # The placeholder goes in without being published; the next publish
        # still records it, after the assistant message that preceded it.
        client.messages += [call, placeholder]
        await to_event_bus(call, cfg)
        checkpoint()
        placeholder["content"] = "42"
        await to_event_bus(placeholder, cfg)
        checkpoint()
        await to_event_bus(placeholder, cfg)  # unchanged: nothing new
        checkpoint()
        return session, offsets, snapshots

    session, offsets, snapshots = asyncio.run(scenario())
    assert offsets[0] < offsets[1] < offsets[2] < offsets[3] == offsets[4]
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier)
    lines = read_lines(session.path)
    assert [ln["type"] for ln in lines] == [
        "session_start",
        "message",
        "message",
        "message",
        "message_update",
    ]
    assert lines[3]["message"]["content"] == json.dumps({"_placeholder": "pending"})
    assert lines[4]["message"]["content"] == "42"


# ── compaction ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compaction_is_recorded_and_the_new_context_points_at_the_file(
    monkeypatch,
    provider,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)
    p = provider(
        {
            # The compressor keeps every entry as it is.
            "You are a context compactor": [completion(content="done")],
            # Everything else is the root conversation, whose system prompt
            # the rebuild folds into the compressed context.
            "": [
                completion(content="thinking", calls=[("lookup", {})]),
                completion(calls=[("compress_context", {})]),
                completion(content="after compaction"),
            ],
        },
    )

    def lookup() -> str:
        """Return a fact."""
        return "the fact is 7"

    client = llm(ROOT_PROMPT)
    handle = start_async_tool_loop(
        client,
        "find the fact",
        {"lookup": lookup},
        loop_id="RootAgent.act",
    )
    assert await handle.result() == "after compaction"
    session = client._unify_transcript
    pointer = session.pointer_line()

    # What the model saw after the restart points at the transcript...
    assert str(session.path) in pointer
    assert not any(pointer in json.dumps(r) for r in p.requests[:-1])
    assert pointer in json.dumps(p.requests[-1])

    end_all_sessions()
    lines = read_lines(session.path)
    kinds = [ln["type"] for ln in lines]
    compaction = lines[kinds.index("compaction")]
    # ...and every message before the compaction is on disk ahead of it.
    before = "\n".join(json.dumps(ln) for ln in lines[: kinds.index("compaction")])
    assert "the fact is 7" in before and "find the fact" in before
    assert compaction["pass"] == 1 and compaction["archived_messages"] >= 4
    assert pointer in json.dumps(compaction["context"])
    # The restarted loop stays in the same session and keeps appending.
    after = lines[kinds.index("compaction") + 1 :]
    assert any(
        ln["type"] == "message" and ln["message"].get("content") == "after compaction"
        for ln in after
    )
    assert kinds.count("session_start") == 1

    index = {line["session"]: line for line in index_lines()}
    assert index[session.id]["compactions"] == 1
    compressor = [ln for ln in index.values() if ln["origin"] == "compress_messages"]
    assert len(compressor) == 1 and compressor[0]["parent"] == session.id


@pytest.mark.asyncio
async def test_compaction_without_the_switch_adds_no_pointer(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", False)

    async def keep(messages, endpoint, **kw):
        return cc.CompressedMessages(
            messages=[
                cc.CompressedMessage(content=json.dumps(m))
                for m in (kw.get("prior_entries") or []) + messages
            ],
        )

    monkeypatch.setattr(cc, "compress_messages", keep)
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    result = await cc.compress_and_rebuild(cc.CompressionState(), msgs, MODEL, {})
    content = result.system_msgs[-1]["content"]
    assert "full history of this session" not in content
    assert content.endswith("retrieve a range of consecutive messages.")
    assert not (home() / "transcripts").exists()


# ── restarts ─────────────────────────────────────────────────────────────────

_PROCESS = textwrap.dedent(
    """
    import asyncio, sys
    from unify import transcripts
    from unify.common._async_tool.event_bus_util import to_event_bus
    from unify.common._async_tool.loop_config import LoopConfig
    from unify.common._async_tool.message_dispatcher import LoopMessageDispatcher
    from unify.settings import SETTINGS

    SETTINGS.UNIFY_TRANSCRIPTS = True

    class Timer:
        def reset(self):
            pass

    class Client:
        system_message = "sys"
        def __init__(self):
            self.messages = []
        def append_messages(self, msgs):
            self.messages += msgs

    async def main(session_id, text):
        client = Client()
        cfg = LoopConfig("Proc", None, [])
        if session_id:
            with transcripts.resume_session(session_id):
                d = LoopMessageDispatcher(client, cfg, Timer())
        else:
            d = LoopMessageDispatcher(client, cfg, Timer())
        await d.append_msgs([{"role": "user", "content": text}])
        print(client._unify_transcript.id)
        transcripts.close(client)

    asyncio.run(main(sys.argv[1], sys.argv[2]))
    """,
)


def run_process(session_id: str, text: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", _PROCESS, session_id, text],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_a_new_process_appends_only_to_the_session_it_names():
    first = run_process("sess-a", "one")
    path = home() / "transcripts" / "sess-a.jsonl"
    before = path.read_bytes()
    second = run_process("sess-a", "two")
    after = path.read_bytes()
    other = run_process("sess-b", "three")
    fresh = run_process("", "four")

    assert first == second == "sess-a" and other == "sess-b"
    assert fresh not in ("sess-a", "sess-b")
    assert after.startswith(before) and len(after) > len(before)
    lines = read_lines(path)
    assert [ln["seq"] for ln in lines] == list(range(len(lines)))
    starts = [ln for ln in lines if ln["type"] == "session_start"]
    assert [s["resumed"] for s in starts] == [False, True]
    texts = [ln["message"]["content"] for ln in lines if ln["type"] == "message"]
    assert texts == ["one", "two"]
    assert "three" not in path.read_text()
    assert len(session_files()) == 3
    assert sorted(i["session"] for i in index_lines()) == sorted(
        ["sess-a", "sess-a", "sess-b", fresh],
    )


def test_two_live_roots_never_share_a_requested_id(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TRANSCRIPTS", True)

    class Client:
        system_message = None

        def __init__(self):
            self.messages = []

    a, b = Client(), Client()
    # Two unrelated loops started from the same resumed context: neither is
    # the other's parent, so both would take the requested id.
    with transcripts.resume_session("shared"):
        ctx_a, ctx_b = contextvars.copy_context(), contextvars.copy_context()
    ctx_a.run(transcripts.attach, a, LoopConfig("A", None, []))
    ctx_b.run(transcripts.attach, b, LoopConfig("B", None, []))
    assert b._unify_transcript.parent_id is None
    assert a._unify_transcript.id == "shared"
    assert b._unify_transcript.id != "shared"
    with pytest.raises(ValueError):
        with transcripts.resume_session("../escape"):
            pass


class _NoTimer:
    def reset(self):
        pass
