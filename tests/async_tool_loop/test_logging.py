from __future__ import annotations

import os
import time
import re
import pytest

from unify.common.async_tool_loop import start_async_tool_loop
from unify.events.event_bus import EVENT_BUS
from unify.common.llm_client import new_llm_client, PendingThinkingLog


@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_nested_logging_hierarchy_labels(llm_config):
    """
    Verify that nested async tool loops emit ToolLoop events with hierarchical
    lineage in payload: `hierarchy` (list[str]) and `hierarchy_label` (str).

    We create an outer loop (loop_id="Outer") whose tool starts an inner loop
    (loop_id="Inner"). We assert that events exist for both levels:
    - hierarchy == ["Outer"]
    - hierarchy == ["Outer", "Inner"] with label "Outer -> Inner"
    """

    # ── inner tool: trivial sync function ──────────────────────────────────
    def inner_tool() -> str:  # noqa: D401
        time.sleep(0.1)
        return "inner-ok"

    # ── outer tool: runs a nested loop to completion and returns its reply ──
    # The loop no longer adopts a handle a tool returns (it stops it and
    # sends the handle's repr, a memory address), so the tool awaits it.
    async def outer_tool() -> str:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are running inside an automated test.\n"
            "1️⃣  Call `inner_tool` (no arguments).\n"
            "2️⃣  Wait for its response.\n"
            "3️⃣  Reply with exactly 'done'.",
        )

        inner = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"inner_tool": inner_tool},
            loop_id="Inner",
            max_steps=10,
            timeout=120,
        )
        return await inner.result()

    outer_tool.__name__ = "outer_tool"
    outer_tool.__qualname__ = "outer_tool"

    # ── top-level loop: uses the outer tool ────────────────────────────────
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are running inside an automated test. Perform the steps exactly:\n"
        "1️⃣  Call `outer_tool` with no arguments.\n"
        "2️⃣  Once it is completed, respond with exactly 'outer done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"outer_tool": outer_tool},
        loop_id="Outer",
        max_steps=10,
        timeout=240,
    )

    # Wait for completion
    final_reply = await handle.result()
    assert final_reply is not None, "Loop should complete with a response"

    # Gather recent ToolLoop events
    events = EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    # Presence checks for hierarchy payloads
    has_outer_only = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and len((evt.payload or {}).get("hierarchy")) == 1
        and (evt.payload or {}).get("hierarchy")[0].startswith("Outer(")
        for evt in events
    )
    has_outer_inner = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and len((evt.payload or {}).get("hierarchy")) == 2
        and (evt.payload or {}).get("hierarchy")[0].startswith("Outer(")
        and (evt.payload or {}).get("hierarchy")[1].startswith("Inner(")
        for evt in events
    )
    has_outer_inner_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Outer\([0-9a-f]{4}\)->Inner\([0-9a-f]{4}\)",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )

    assert has_outer_only, "No ToolLoop event recorded with hierarchy ['Outer']"
    assert (
        has_outer_inner
    ), "No ToolLoop event recorded with hierarchy ['Outer', 'Inner']"
    assert (
        has_outer_inner_label
    ), "No ToolLoop event recorded with hierarchy_label 'Outer -> Inner'"


@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_single_loop_logging_hierarchy_label(llm_config):
    """
    Verify that a single (non-nested) async tool loop emits ToolLoop events
    with a flat hierarchy and label equal to its loop_id.

    We start a solo loop with loop_id="Solo" and a trivial tool.
    Assertions:
    - hierarchy == ["Solo"] exists
    - hierarchy_label == "Solo" exists
    - no event exists with hierarchy beginning ["Solo", ...] (i.e., nested)
    """

    def noop_tool() -> str:  # noqa: D401
        return "ok"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "1️⃣  Call `noop_tool`. 2️⃣ Then reply exactly 'done'.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"noop_tool": noop_tool},
        loop_id="Solo",
        max_steps=10,
        timeout=120,
    )

    final_reply = await handle.result()
    assert final_reply is not None, "Loop should complete with a response"

    events = EVENT_BUS.search(filter="type == 'ToolLoop'", limit=200)

    has_solo = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and len((evt.payload or {}).get("hierarchy")) == 1
        and (evt.payload or {}).get("hierarchy")[0].startswith("Solo(")
        for evt in events
    )
    has_solo_label = any(
        isinstance((evt.payload or {}).get("hierarchy_label"), str)
        and re.fullmatch(
            r"Solo\([0-9a-f]{4}\)",
            (evt.payload or {}).get("hierarchy_label"),
        )
        for evt in events
    )
    has_nested_under_solo = any(
        isinstance((evt.payload or {}).get("hierarchy"), list)
        and len((evt.payload or {}).get("hierarchy")) > 1
        and (evt.payload or {}).get("hierarchy")[0].startswith("Solo(")
        for evt in events
    )

    assert has_solo, "No ToolLoop event recorded with hierarchy ['Solo']"
    assert has_solo_label, "No ToolLoop event recorded with hierarchy_label 'Solo'"
    assert not has_nested_under_solo, "Unexpected nested hierarchy found under 'Solo'"


@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_litellm_logs_are_suppressed(llm_config, caplog):
    """
    Verify that LiteLLM logs are suppressed by our logging configuration.

    LiteLLM creates loggers like 'LiteLLM', 'LiteLLM Proxy', 'LiteLLM Router'.
    These should be muted to WARNING level so INFO logs don't pollute output.
    This test catches regressions if LiteLLM changes logger names or adds new ones.
    """
    import logging

    # Capture all logs at DEBUG level to ensure we catch everything
    caplog.set_level(logging.DEBUG)

    def noop_tool() -> str:
        return "ok"

    client = new_llm_client(**llm_config)
    client.set_system_message("Call noop_tool, then reply 'done'.")

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"noop_tool": noop_tool},
        loop_id="LiteLLMTest",
        max_steps=5,
        timeout=60,
    )

    await handle.result()

    # Check that no log records come from LiteLLM loggers
    litellm_logs = [
        record for record in caplog.records if "litellm" in record.name.lower()
    ]

    assert not litellm_logs, (
        f"Found {len(litellm_logs)} log(s) from LiteLLM loggers that should be suppressed. "
        f"Logger names: {sorted(set(r.name for r in litellm_logs))}"
    )


@pytest.mark.asyncio
async def test_inline_log_file_paths(unify_logs, tmp_path):
    """
    Verify that when UNILLM_LOG_DIR is set, the async tool loop emits a
    combined "LLM thinking… → /path" line that merges the thinking indicator
    with the log file path into a single line per LLM call.

    The test:
    1. Configures a temporary UNILLM_LOG_DIR
    2. Runs a single-tool loop with loop_id="LogFileTest"
    3. Asserts a 🧠 record carries "LLM thinking…", the lineage label, and
       a "→ …/path.txt" reference — all in one line
    4. Asserts the referenced file actually exists on disk
    5. Asserts no separate 📝 filepath lines exist (they are combined now)

    The model is scripted (``tests/cache_discipline_helpers``), so the call
    runs with the LLM cache off: its pending file is ``{base}.pending.txt``
    (``{base}.cache_pending.txt`` with the cache on).
    """
    from tests import cache_discipline_helpers as h
    import unillm.logger as unillm_logger

    log_dir = tmp_path / "unillm_logs"
    log_dir.mkdir()
    old_env = os.environ.get("UNILLM_LOG_DIR")
    unillm_logger.configure_log_dir(str(log_dir))

    try:

        def noop_tool() -> str:
            return "ok"

        replies = [
            lambda: h.completion(calls=[("noop_tool", {})]),
            lambda: h.completion(content="done"),
        ]
        with h.scripted(replies):
            handle = start_async_tool_loop(
                client=h.new_client("Call noop_tool, then reply 'done'."),
                message="start",
                tools={"noop_tool": noop_tool},
                loop_id="LogFileTest",
                max_steps=5,
                timeout=60,
            )
            await handle.result()

        logged_lines = unify_logs.text.splitlines()

        combined_lines = [
            line
            for line in logged_lines
            if "🧠" in line
            and "LLM thinking" in line
            and "→" in line
            and "LogFileTest" in line
        ]

        assert combined_lines, (
            "No combined 'LLM thinking… → /path' lines found. "
            f"Logged lines containing LogFileTest: "
            f"{[l for l in logged_lines if 'LogFileTest' in l]}"
        )

        for line in combined_lines:
            match = re.search(r"→ (.+\.txt)", line)
            assert match, f"Could not extract .txt path from log line: {line}"
            path_str = match.group(1)
            assert (
                str(log_dir) in path_str
            ), f"Log file {path_str} is not under expected dir {log_dir}"

            # The pending path ({base}.pending.txt, or .cache_pending.txt with
            # the cache on) is renamed after the call completes. Verify a
            # finalized file with the same base exists.
            base = re.split(r"\.(?:cache_)?pending\.", os.path.basename(path_str))[0]
            finalized = [
                f
                for f in os.listdir(log_dir)
                if f.startswith(base) and f != os.path.basename(path_str)
            ]
            assert finalized, (
                f"No finalized log file found for base '{base}' in {log_dir}. "
                f"Files: {os.listdir(log_dir)}"
            )

        separate_filepath_lines = [
            line
            for line in logged_lines
            if "📝" in line and "LogFileTest" in line and "→" in line
        ]
        assert not separate_filepath_lines, (
            "Found separate 📝 filepath lines — these should be combined with "
            f"the thinking line now: {separate_filepath_lines}"
        )

    finally:
        if old_env is not None:
            os.environ["UNILLM_LOG_DIR"] = old_env
        else:
            os.environ.pop("UNILLM_LOG_DIR", None)
        unillm_logger.configure_log_dir(old_env)


@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_thinking_log_fallback_without_log_dir(llm_config, unify_logs):
    """
    When UNILLM_LOG_DIR is NOT set, the async tool loop should still emit
    plain "🧠 LLM thinking…" lines (without a filepath) as a fallback.
    """
    import unillm.logger as unillm_logger

    old_env = os.environ.get("UNILLM_LOG_DIR")
    os.environ.pop("UNILLM_LOG_DIR", None)
    unillm_logger.configure_log_dir(None)

    try:

        def noop_tool() -> str:
            return "ok"

        client = new_llm_client(**llm_config)
        client.set_system_message("Call noop_tool, then reply 'done'.")

        handle = start_async_tool_loop(
            client=client,
            message="start",
            tools={"noop_tool": noop_tool},
            loop_id="FallbackTest",
            max_steps=5,
            timeout=60,
        )

        await handle.result()

        logged_lines = unify_logs.text.splitlines()

        thinking_lines = [
            line
            for line in logged_lines
            if "🧠" in line and "LLM thinking" in line and "FallbackTest" in line
        ]

        assert thinking_lines, (
            "No 'LLM thinking…' lines found when UNILLM_LOG_DIR is unset. "
            f"Logged lines containing FallbackTest: "
            f"{[l for l in logged_lines if 'FallbackTest' in l]}"
        )

        for line in thinking_lines:
            assert (
                "→" not in line
            ), f"Fallback thinking line should NOT contain '→' filepath: {line}"

    finally:
        if old_env is not None:
            os.environ["UNILLM_LOG_DIR"] = old_env
        unillm_logger.configure_log_dir(old_env)


# ── PendingThinkingLog unit tests ─────────────────────────────────────────


class TestPendingThinkingLog:

    def test_combined_line_with_context(self, unify_logs):
        """When a pending path arrives and thinking context is set,
        a single combined '🧠 ... LLM thinking…(suffix) → /path' line is emitted."""
        from pathlib import Path

        log = PendingThinkingLog("FastBrain")
        log.set_thinking_context(" (test)")
        log.on_pending_path(Path("/tmp/fake.cache_pending.txt"))

        out = unify_logs.text
        assert "🧠" in out
        assert "[FastBrain]" in out
        assert "LLM thinking…" in out
        assert "(test)" in out
        assert "generation_id" not in out
        assert "source_id" not in out
        assert "→ /tmp/fake.cache_pending.txt" in out
        lines = [l for l in out.splitlines() if "FastBrain" in l]
        assert len(lines) == 1

    def test_fallback_without_log_dir(self, unify_logs):
        """When no pending path arrives, emit_fallback produces a plain thinking line."""
        log = PendingThinkingLog("ConversationManager")
        log.set_thinking_context(" (UserWebcamStarted)")
        log.emit_fallback()

        out = unify_logs.text
        assert "🧠" in out
        assert "[ConversationManager]" in out
        assert "LLM thinking…" in out
        assert "(UserWebcamStarted)" in out
        assert "→" not in out

    def test_fallback_suppressed_after_pending(self, unify_logs):
        """If pending callback already fired, emit_fallback is a no-op."""
        from pathlib import Path

        log = PendingThinkingLog("ProactiveSpeech")
        log.on_pending_path(Path("/tmp/fake.txt"))

        assert len(unify_logs.records) == 1, "the pending path should log once"
        unify_logs.clear()

        log.emit_fallback()
        assert unify_logs.records == []

    def test_no_context_produces_generic_line(self, unify_logs):
        """Without set_thinking_context, the combined line has no suffix."""
        from pathlib import Path

        log = PendingThinkingLog("ProactiveSpeech")
        log.on_pending_path(Path("/tmp/fake.txt"))

        out = unify_logs.text
        assert "LLM thinking… →" in out
        assert "ProactiveSpeech" in out

    @pytest.mark.llm_call
    def test_new_llm_client_attaches_pending_log(self):
        """new_llm_client with origin should attach _pending_thinking_log."""
        client = new_llm_client(origin="TestOrigin")
        assert hasattr(client, "_pending_thinking_log")
        assert isinstance(client._pending_thinking_log, PendingThinkingLog)
