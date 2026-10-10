"""Symbolic: the actor gets bash cells, ``read_file`` and ``grep`` in the workspace sandbox.

Nothing here reaches a model; the tools are the actor's real ones. With the
switch off the actor's tools and the schema the model sees are as shipped.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    SSH_SECRET,
    STATE_SECRET,
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

# ── helpers ─────────────────────────────────────────────────────────────────


def _schema(actor, name):
    spec = actor.get_tools("act")[name]
    fn = spec.fn if isinstance(spec, ToolSpec) else spec
    return fn, llmh.method_to_schema(fn, name)


# ── read_file and grep ───────────────────────────────────────────────────────


def test_read_file_reads_numbered_ranges_and_refuses_hidden_paths(world):
    policy = sandbox.build_policy(fresh=True)
    home, state = world["home"], world["state"]
    out = file_tools.read_file("data.txt", 3, 4, policy=policy)
    assert out["content"] == "     3\tline 3\n     4\tline 4\n"
    assert out["total_lines"] == 10 and out["end"] == 4
    # Outside the allowlisted root: the sandbox does not show it, so neither
    # does the harness's own file tool.
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        file_tools.read_file(str(home / "plain.txt"), policy=policy)
    assert refused.value.rule == "root-allowlist"
    out = file_tools.read_file(str(state / "transcripts" / "s.jsonl"), policy=policy)
    assert '"seq": 0' in out["content"]
    (world["workspace"] / "link").symlink_to(home / ".ssh" / "id_rsa")
    # The raw store is not mounted into cells (they use the functions/guidance
    # API), so the harness's file tools refuse it too, with its -wal and -shm.
    store = state / "store.sqlite"
    for suffix in ("-wal", "-shm"):
        store.with_name(store.name + suffix).write_text(STATE_SECRET)
    refusals = {
        str(store): "mask-unify-state",
        str(store) + "-wal": "mask-unify-state",
        str(store) + "-shm": "mask-unify-state",
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
@pytest.mark.parametrize("engine", ["ripgrep", "python-sandboxed"])
@pytest.mark.asyncio
async def test_grep_never_searches_hidden_paths(world, monkeypatch, engine):
    if engine == "python-sandboxed":
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
    # The home is outside the allowlisted root: not searchable at all.
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await file_tools.grep("needle", str(home), policy=policy)
    assert refused.value.rule == "root-allowlist"
    # Inside the workspace (what a cell may read anyway) it searches as before.
    project = world["workspace"] / "project"
    project.mkdir()
    (project / "notes.txt").write_text("alpha\nneedle one\nbeta\n")
    (project / "more.txt").write_text(f"needle two {SSH_SECRET}\n")
    out = await file_tools.grep("needle", str(project), policy=policy)
    assert out["engine"] == engine
    assert sorted(h.split(":", 2)[2] for h in out["hits"]) == [
        "needle one",
        f"needle two {SSH_SECRET}",
    ]
    for secret in (ENV_SECRET, "PRIVATE-KEY"):
        found = await file_tools.grep(secret, str(project), policy=policy)
        assert all("more.txt" in h for h in found["hits"]), found
    out = await file_tools.grep("line", "data.txt", max_hits=3, policy=policy)
    assert len(out["hits"]) == 3 and out["truncated"]
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await file_tools.grep("x", str(home / ".ssh"), policy=policy)
    assert refused.value.rule == "mask-credentials"
    # The raw store is not mounted into cells, so grep refuses it as well.
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await file_tools.grep("f", str(world["state"] / "store.sqlite"), policy=policy)
    assert refused.value.rule == "mask-unify-state"


def _without_ripgrep(monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(
        file_tools.shutil,
        "which",
        lambda name: None if name == "rg" else real_which(name),
    )


def _spy_on_regex_compiles(monkeypatch) -> list:
    """Every pattern ``re`` compiles in this (the harness's) process."""
    seen: list = []
    real = re._compile

    def spy(pattern, flags):
        seen.append(pattern)
        return real(pattern, flags)

    monkeypatch.setattr(re, "_compile", spy)
    return seen


def _processes_naming(token: str) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                if token.encode() in fh.read():
                    found.append(int(entry))
        except OSError:
            continue
    return found


@needs_bwrap
@pytest.mark.asyncio
async def test_grep_without_ripgrep_matches_in_the_sandbox_not_the_harness(
    world,
    monkeypatch,
):
    """The fallback's model-chosen regex is compiled and run by a sandboxed
    child; the harness only parses its hits and checks each path."""
    _without_ripgrep(monkeypatch)
    policy = sandbox.build_policy(fresh=True)
    project = world["workspace"] / "project"
    project.mkdir()
    (project / "a.txt").write_text("x\nneedle-7f3e here\n")
    (project / "b.txt").write_text("needle-7f3e again\n")
    pattern = r"needle-7f3e\s\w+"
    try:
        re.compile("(unclosed")
    except re.error as exc:
        expected = f"Invalid regular expression '(unclosed': {exc}"
    seen = _spy_on_regex_compiles(monkeypatch)
    out = await file_tools.grep(pattern, str(project), policy=policy)
    assert out["engine"] == "python-sandboxed"
    assert out["hits"] == [
        f"{project / 'a.txt'}:2:needle-7f3e here",
        f"{project / 'b.txt'}:1:needle-7f3e again",
    ]
    assert out["truncated"] is False
    assert pattern not in seen
    assert not hasattr(file_tools, "_grep_python")
    # An invalid pattern: the child's compile error, the same message as before.
    with pytest.raises(ToolInputError) as bad:
        await file_tools.grep("(unclosed", str(project), policy=policy)
    assert str(bad.value) == expected
    assert "(unclosed" not in seen
    # A hit in a path the policy refuses is dropped, whatever the child saw.
    real_violation = policy.readable_violation
    monkeypatch.setattr(
        policy,
        "readable_violation",
        lambda path: (
            ("mask-test", "refused by the test")
            if Path(path).name == "b.txt"
            else real_violation(path)
        ),
    )
    out = await file_tools.grep("needle-7f3e", str(project), policy=policy)
    assert [h.split(":", 1)[0] for h in out["hits"]] == [str(project / "a.txt")]


@needs_bwrap
@pytest.mark.asyncio
async def test_grep_fallback_times_out_on_catastrophic_backtracking(
    world,
    monkeypatch,
):
    """``(a+)+$`` on ``aaaa...b`` backtracks for ~2^60 steps: the child is
    killed at the wall limit, the harness gets a clear error, and its event
    loop keeps running throughout."""
    _without_ripgrep(monkeypatch)
    monkeypatch.setattr(file_tools, "GREP_TIMEOUT_S", 2.0)
    policy = sandbox.build_policy(fresh=True)
    token = f"redos-{uuid.uuid4().hex[:12]}"
    target = world["workspace"] / token
    target.mkdir()
    (target / "long.txt").write_text("a" * 60 + "b\n")
    seen = _spy_on_regex_compiles(monkeypatch)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    beat = asyncio.create_task(ticker())
    started = time.monotonic()
    try:
        with pytest.raises(ToolInputError, match="timed out after 2s"):
            await file_tools.grep(r"(a+)+$", str(target), policy=policy)
    finally:
        beat.cancel()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0 + 8.0, elapsed
    assert ticks >= 20, ticks  # ~40 expected over 2 s; never blocked
    assert r"(a+)+$" not in seen
    # Verified termination: no process still names the search.
    deadline = time.monotonic() + 5.0
    while _processes_naming(token) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert _processes_naming(token) == []


@pytest.mark.asyncio
async def test_grep_without_bubblewrap_refuses_and_never_matches_in_process(
    world,
    monkeypatch,
):
    _without_ripgrep(monkeypatch)
    monkeypatch.setattr(sandbox, "bwrap_path", lambda: None)
    policy = sandbox.build_policy(fresh=True)
    seen = _spy_on_regex_compiles(monkeypatch)
    with pytest.raises(sandbox.SandboxRefusal) as refused:
        await file_tools.grep("line 1[0-9]?", "data.txt", policy=policy)
    assert refused.value.rule == "sandbox-required"
    assert "line 1[0-9]?" not in seen


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


# ── under the cache discipline ─────────────────────────────────────────────
