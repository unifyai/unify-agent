"""Symbolic: under ``UNIFY_WORKSPACE=sandboxed`` the actor gets bash cells, ``read_file`` and ``grep``.

Nothing here reaches a model; the tools are the actor's real ones. With the
switch off the actor's tools and the schema the model sees are as shipped.
"""

from __future__ import annotations

import inspect
import json
import shutil

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    SSH_SECRET,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import file_tools
from unify.actor.execution.types import parts_to_text
from unify.common import llm_helpers as llmh
from unify.common.tool_errors import ToolInputError
from unify.common.tool_spec import ToolSpec
from unify.settings import SETTINGS

# ── off: exactly as shipped ─────────────────────────────────────────────────


def _schema(actor, name):
    spec = actor.get_tools("act")[name]
    fn = spec.fn if isinstance(spec, ToolSpec) else spec
    return fn, llmh.method_to_schema(fn, name)


@pytest.mark.asyncio
async def test_off_the_actor_has_no_shell_language_and_no_file_tools(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "")
    actor = CodeActActor(environments=[])
    try:
        tools = set(actor.get_tools("act"))
        assert not {"read_file", "grep"} & tools
        fn, schema = _schema(actor, "execute_code")
        params = schema["function"]["parameters"]["properties"]
        assert "language" not in params and "_language" not in params
        assert "there is no shell cell" in schema["function"]["description"]
        # The hidden parameter changes nothing the model sees: the schema is
        # the one the function had without it.
        sig = inspect.signature(fn)
        fn.__signature__ = sig.replace(
            parameters=[p for p in sig.parameters.values() if p.name != "_language"],
        )
        try:
            assert llmh.method_to_schema(fn, "execute_code") == schema
        finally:
            del fn.__signature__
    finally:
        await actor.close()


# ── read_file and grep ───────────────────────────────────────────────────────


def test_read_file_reads_numbered_ranges_and_refuses_hidden_paths(world):
    policy = sandbox.build_policy(fresh=True)
    home, state = world["home"], world["state"]
    out = file_tools.read_file("data.txt", 3, 4, policy=policy)
    assert out["content"] == "     3\tline 3\n     4\tline 4\n"
    assert out["total_lines"] == 10 and out["end"] == 4
    out = file_tools.read_file(str(home / "plain.txt"), policy=policy)
    assert "readable outside" in out["content"]
    out = file_tools.read_file(str(state / "transcripts" / "s.jsonl"), policy=policy)
    assert '"seq": 0' in out["content"]
    (world["workspace"] / "link").symlink_to(home / ".ssh" / "id_rsa")
    refusals = {
        str(home / ".ssh" / "id_rsa"): "mask-credentials",
        "link": "mask-credentials",
        str(home / ".env"): "mask-env-file",
        str(home / "project" / ".env"): "mask-env-file",
        str(state / "logs" / "unify.log"): "mask-unify-state",
        "/proc/self/environ": "mask-proc",
        str(state / "workspace"): "regular-files-only",
    }
    for path, rule in refusals.items():
        with pytest.raises(sandbox.SandboxRefusal) as refused:
            file_tools.read_file(path, policy=policy)
        assert refused.value.rule == rule, path
        assert f"rule `{rule}`" in refused.value.as_tool_result()


@needs_bwrap
@pytest.mark.parametrize("engine", ["ripgrep", "python"])
@pytest.mark.asyncio
async def test_grep_never_searches_hidden_paths(world, monkeypatch, engine):
    if engine == "python":
        real_which = shutil.which
        monkeypatch.setattr(
            file_tools.shutil,
            "which",
            lambda name: None if name == "rg" else real_which(name),
        )
    elif shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    policy = sandbox.build_policy(fresh=True)
    home = world["home"]
    (home / "project" / "more.txt").write_text(f"needle two {SSH_SECRET}\n")
    out = await file_tools.grep("needle", str(home), policy=policy)
    assert out["engine"] == engine
    assert sorted(h.split(":", 2)[2] for h in out["hits"]) == [
        "needle one",
        f"needle two {SSH_SECRET}",
    ]
    for secret in (ENV_SECRET, "PRIVATE-KEY"):
        found = await file_tools.grep(secret, str(home), policy=policy)
        assert all("more.txt" in h for h in found["hits"]), found
    out = await file_tools.grep("line", "data.txt", max_hits=3, policy=policy)
    assert len(out["hits"]) == 3 and out["truncated"]
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await file_tools.grep("x", str(home / ".ssh"), policy=policy)
    assert refused.value.rule == "mask-credentials"


# ── the actor's tools ────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_on_the_actor_offers_bash_read_file_and_grep(world):
    actor = CodeActActor(environments=[])
    try:
        tools = actor.get_tools("act")
        assert {"read_file", "grep", "execute_code"} <= set(tools)
        fn, schema = _schema(actor, "execute_code")
        props = schema["function"]["parameters"]["properties"]
        assert "language" in props and "_language" not in props
        desc = schema["function"]["description"]
        assert "Python or bash" in desc and "there is no network" in desc
        assert "there is no shell cell" not in desc
        execute_code = tools["execute_code"]
        await execute_code("set a variable", "export N=5", language="bash")
        out = await execute_code("read it", 'echo "n=$N"', language="bash")
        assert parts_to_text(out.stdout).strip() == "n=5" and out.result == 0
        out = await execute_code("python still works", "6 * 7")
        assert out.result == 42
        with pytest.raises(ToolInputError, match="Unsupported language"):
            await execute_code("ruby", "puts 1", language="ruby")
        read = await tools["read_file"].fn("data.txt", 1, 1)
        assert read["content"] == "     1\tline 1\n"
        found = await tools["grep"].fn("line 1", ".")
        assert any(h.endswith("line 10") for h in found["hits"])
        json.dumps(read), json.dumps(found)
    finally:
        await actor.close()
