from __future__ import annotations

import logging

from unify.legacy.conversation_manager.conversation_manager import (
    _log_slow_brain_single_shot_failure,
)
from unify.logger import LOGGER


def test_debug_slow_brain_single_shot_failure_logs_traceback(caplog) -> None:
    """With the unify logger at DEBUG, a single-shot failure logs its traceback."""
    LOGGER.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="unify")

    try:
        try:
            raise TypeError("string indices must be integers, not 'str'")
        except TypeError:
            _log_slow_brain_single_shot_failure(
                run_id="llmrun-000001",
                request_id="llmreq-000001",
                origin_event_name="UnifyMessageReceived",
                message_count=4,
                tool_count=12,
                state_chars=3456,
            )
    finally:
        LOGGER.removeHandler(caplog.handler)

    assert "Slow-brain single-shot failed" in caplog.text
    assert "run_id=llmrun-000001" in caplog.text
    assert "request_id=llmreq-000001" in caplog.text
    assert "origin_event=UnifyMessageReceived" in caplog.text
    assert "Slow-brain single-shot traceback text" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert any(record.exc_info for record in caplog.records)
