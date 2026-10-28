"""How the loop tallies refused calls, and when it stops.

A refusal is how a caller converges on an argspec: it is told what is wrong and
tries again. Counting refusals the way unexpected exceptions are counted would
abort exactly that. So refusals end the loop only on repetition — the same call
sent again unchanged, or the same complaint earned while something irrelevant is
varied around it.
"""

from unify.common._async_tool.loop import (
    _LoopToolFailureTracker,
    ToolLoopRuntimeState,
)


def _tracker() -> _LoopToolFailureTracker:
    return _LoopToolFailureTracker(ToolLoopRuntimeState())


def _refuse(tracker, *, args, message="Cannot use state_mode='stateless'."):
    return tracker.note_refusal(
        tool_name="execute_code",
        args=args,
        message=message,
    )


# ── working out an argspec is free ─────────────────────────────────────────


def test_distinct_attempts_are_never_stopped():
    """Varying the argument the refusal is about is progress, however long it takes."""
    tracker = _tracker()
    for attempt in range(40):
        stop = _refuse(
            tracker,
            args={"session_id": attempt},
            message=f"session_id {attempt} does not exist.",
        )
        assert stop is None, f"stopped on attempt {attempt}: {stop}"
    assert tracker.stop_reason() is None


def test_refusals_do_not_consume_the_unexpected_failure_budget():
    """A refusal is not a fault, so it must not spend the budget for real ones."""
    tracker = _tracker()
    for attempt in range(10):
        _refuse(tracker, args={"session_id": attempt}, message=f"no {attempt}")
    assert tracker.current_failures == 0
    assert not tracker.has_exceeded_failures()


# ── repetition is not ──────────────────────────────────────────────────────


def test_the_same_call_refused_three_times_stops_the_loop():
    """Sending back a byte-identical call means the refusal was not read."""
    tracker = _tracker()
    args = {"state_mode": "stateless", "session_id": 0, "session_name": ""}
    assert _refuse(tracker, args=args) is None
    assert _refuse(tracker, args=args) is None
    stop = _refuse(tracker, args=args)
    assert stop is not None
    assert "same arguments" in stop
    assert "execute_code" in stop
    # The reason survives for the guard that ends the loop later in the call.
    assert tracker.stop_reason() == stop


def test_the_same_complaint_stops_the_loop_even_as_arguments_vary():
    """Varying a field the refusal is not about is not converging on anything."""
    tracker = _tracker()
    stop = None
    for attempt in range(_LoopToolFailureTracker.SAME_COMPLAINT_LIMIT):
        stop = _refuse(
            tracker,
            args={"session_name": f"attempt_{attempt}"},
            message="Cannot use state_mode='stateless' with a session.",
        )
    assert stop is not None
    assert "same reason" in stop


def test_identical_calls_stop_sooner_than_varied_ones():
    """The two limits are ordered: exact repetition is the clearer signal."""
    assert (
        _LoopToolFailureTracker.IDENTICAL_CALL_LIMIT
        < _LoopToolFailureTracker.SAME_COMPLAINT_LIMIT
    )


def test_repetition_is_counted_per_tool():
    """One tool's refusals say nothing about another's."""
    state = ToolLoopRuntimeState()
    tracker = _LoopToolFailureTracker(state)
    args = {"session_id": 0}
    for tool in ("execute_code", "execute_function", "run_query"):
        for _ in range(_LoopToolFailureTracker.IDENTICAL_CALL_LIMIT - 1):
            assert (
                tracker.note_refusal(
                    tool_name=tool,
                    args=args,
                    message="same complaint",
                )
                is None
            )
    assert tracker.stop_reason() is None


# ── unexpected exceptions keep their own, tighter budget ───────────────────


def test_unexpected_failures_still_stop_after_the_flat_limit():
    tracker = _tracker()
    for _ in range(3):
        tracker.increment_failures()
    assert tracker.stop_reason() == "Aborted after too many consecutive tool failures."


def test_a_success_clears_the_unexpected_failure_run():
    tracker = _tracker()
    tracker.increment_failures()
    tracker.increment_failures()
    tracker.reset_failures()
    assert tracker.current_failures == 0
    assert tracker.stop_reason() is None


# ── malformed arguments are refusals, so repetition still ends the loop ────
#
# The loop cannot parse arguments that are not valid JSON, so it hands the call
# back to the model as a refusal and moves on. That `continue` is only safe
# because an unchanged retry is counted here — otherwise a model stuck emitting
# the same truncated payload would spin forever.

# The real shape, shortened: a valid prefix, then whitespace, never closed.
TRUNCATED_ARGS = (
    '{"function_name":"primitives.workspace_email.list_messages",'
    '"call_kwargs":{"max_results":' + "\n  " * 200
)
MALFORMED_COMPLAINT = (
    "⚠️ Error: the arguments for 'execute_function' were not valid JSON "
    "(Expecting value: line 1 column 89 (char 88))."
)


def test_an_unchanged_truncated_payload_stops_the_loop():
    tracker = _tracker()
    stops = [
        tracker.note_refusal(
            tool_name="execute_function",
            args=TRUNCATED_ARGS,
            message=MALFORMED_COMPLAINT,
        )
        for _ in range(_LoopToolFailureTracker.IDENTICAL_CALL_LIMIT)
    ]

    assert stops[0] is None, "the first truncation is worth retrying"
    assert stops[-1] is not None, "an unchanged retry must not spin"
    assert tracker.stop_reason() is not None


def test_a_repaired_payload_is_not_penalised():
    """Emitting complete arguments after a truncation is exactly the recovery."""
    tracker = _tracker()
    tracker.note_refusal(
        tool_name="execute_function",
        args=TRUNCATED_ARGS,
        message=MALFORMED_COMPLAINT,
    )

    assert tracker.stop_reason() is None
    assert tracker.current_failures == 0
