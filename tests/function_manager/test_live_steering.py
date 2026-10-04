"""Deterministic contracts for steering a block while it is still running.

Covers the four parts that make a correction reach live work: the AST probes,
the position the runtime tracks, the idempotency cache that lets a retry skip
what already happened, and the retry loop that splices a patch into the source
and runs it again.

Symbolic throughout, so a failure here is a regression in the machinery
rather than a judgement call: the patch is supplied directly, which pins what
the mechanism does with a given correction. Whether a real model writes a
usable correction in the first place is a different question, covered against
a live LLM in ``test_live_steering_eval``.
"""

from __future__ import annotations

import ast
import asyncio

import pytest

from unify.actor.execution.session import PythonExecutionSession
from unify.function_manager.steering import (
    AROUND_CP_FN,
    CP_FN,
    ControlledInterruption,
    ExecutionStopped,
    InterruptionRequest,
    MemoisedDispatch,
    Patch,
    SteeringSession,
    bind_session,
    current_session,
    instrument,
    make_cache_key,
    restore_session,
    run_with_steering,
    splice_patch,
)

NS = {"primitives"}


def built(source: str) -> str:
    return ast.unparse(instrument(ast.parse(source), tool_namespaces=set(NS)))


async def run_block(source: str, session: SteeringSession, namespace: dict) -> object:
    """Compile an instrumented block into the async wrapper the sandbox uses."""
    body = built(source)
    wrapped = "async def __w():\n" + "".join(f"    {ln}\n" for ln in body.splitlines())
    exec(wrapped, namespace)
    return await namespace["__w"]()


async def _stop_author(*, interjections, session):
    return InterruptionRequest(reason=interjections[0], stop=True)


# ── probes ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_checkpoint_fires_a_stop_request():
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("abandon this")
    session = SteeringSession(interject_q=queue, patch_author=_stop_author)

    with pytest.raises(ControlledInterruption, match="abandon this"):
        await session.cp("before a bare statement")


def test_loop_body_is_probed_before_its_side_effect():
    out = built("for v in vendors:\n    await primitives.comms.send(v)\n")
    body = out.split("for v in vendors:\n")[1]
    assert body.index("increment_loop_iteration") < body.index("primitives.comms.send")
    assert body.index(CP_FN) < body.index("primitives.comms.send")


def test_loop_context_is_opened_and_closed():
    out = built("for v in vs:\n    await primitives.a.b(v)\n")
    assert "start_loop_context" in out
    assert "end_loop_context" in out
    # In a finally, so an exception leaving the loop cannot strand the context.
    assert "finally:" in out


def test_branches_are_distinguished():
    out = built(
        "if x:\n    await primitives.a.b()\nelse:\n    await primitives.c.d()\n",
    )
    assert "push_path_context('if_1_true')" in out
    assert "push_path_context('if_1_false')" in out


def test_try_blocks_are_tracked_in_every_arm():
    out = built(
        "try:\n"
        "    await primitives.a.b()\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    pass\n"
        "finally:\n"
        "    pass\n",
    )
    for arm in ("try_1_body", "try_1_except_0", "try_1_else", "try_1_finally"):
        assert arm in out, arm


def test_await_inside_an_expression_is_bracketed():
    """One statement, many dispatches — no statement probe can see them."""
    out = built("[await primitives.comms.send(v) for v in vs]")
    assert AROUND_CP_FN in out


def test_non_tool_awaits_are_left_alone():
    out = built("await asyncio.sleep(1)")
    assert AROUND_CP_FN not in out


def test_async_function_entry_is_probed():
    out = built("async def helper():\n    await primitives.a.b()\n")
    assert "Enter function: helper" in out
    assert "_int('helper')" in out


def test_sync_function_gets_position_probes_but_cannot_suspend():
    out = built("def helper():\n    for i in xs:\n        pass\n")
    inner = out.split("def helper")[1]
    assert "increment_loop_iteration" in inner
    assert f"await {CP_FN}" not in inner


def test_sync_function_gets_a_sync_interrupt_probe():
    """Raising needs no await, so sync frames check for a pending correction
    at entry and at every iteration — without gaining an await they cannot
    make."""
    out = built("def helper():\n    for i in xs:\n        pass\n")
    inner = out.split("def helper")[1]
    assert inner.count("_int_s('helper')") == 2
    assert "await _int_s" not in inner


def test_class_bodies_are_left_alone():
    out = built("class C:\n    for i in xs:\n        pass\n")
    assert CP_FN not in out.split("class C")[1]


@pytest.mark.parametrize(
    "source",
    [
        "for v in vs:\n    await primitives.a.b(v)\n",
        "if x:\n    await primitives.a.b()\nelse:\n    pass\n",
        "async def f():\n    for i in xs:\n        if i:\n            await primitives.a.b(i)\n",
        "try:\n    await primitives.a.b()\nfinally:\n    pass\n",
    ],
)
def test_instrumented_code_compiles_in_the_async_wrapper(source):
    out = built(source)
    compile(
        "async def __w():\n" + "".join(f"    {ln}\n" for ln in out.splitlines()),
        "<test>",
        "exec",
    )


# ── cache identity ─────────────────────────────────────────────────────────
def test_repeated_identical_calls_stay_distinct():
    """Two deliberate sends of the same thing are two effects, not one."""
    session = SteeringSession()
    first = make_cache_key(session, "primitives.comms.send", ("a",), {})
    second = make_cache_key(session, "primitives.comms.send", ("a",), {})
    assert first != second


def test_identity_survives_a_change_of_position():
    """The correction that matters most restructures control flow.

    Narrowing a loop with a filter puts every surviving call inside a new
    branch. Keying on execution position would change their keys and re-send
    everything already sent, which is the failure this identity avoids.
    """
    session = SteeringSession()
    session.runtime.start_loop_context("for_1")
    session.runtime.increment_loop_iteration("for_1")
    positioned = make_cache_key(session, "primitives.comms.send", ("a",), {})

    replayed = SteeringSession()
    replayed.runtime.start_loop_context("for_1")
    replayed.runtime.increment_loop_iteration("for_1")
    replayed.runtime.push_path_context("if_1_true")
    assert make_cache_key(replayed, "primitives.comms.send", ("a",), {}) == positioned


def test_occurrences_reset_for_a_retry():
    session = SteeringSession()
    make_cache_key(session, "t", ("a",), {})
    session.runtime.reset_position()
    assert make_cache_key(session, "t", ("a",), {})[-1] == 0


@pytest.mark.asyncio
async def test_completed_calls_name_their_arguments():
    """A correction is written against what was sent, not just that it was."""
    session = SteeringSession()

    class _Comms:
        async def send(self, to, *, urgent=False):
            return to

    class _P:
        comms = _Comms()

    prims = MemoisedDispatch(_P(), session)
    await prims.comms.send("eu-alpha")
    await prims.comms.send("us-beta", urgent=True)
    assert session.cache.completed_calls() == [
        "comms.send('eu-alpha')",
        "comms.send('us-beta', urgent=True)",
    ]


def test_invalidation_is_by_prefix():
    session = SteeringSession()
    session.cache.put(("k1",), 1, tool="primitives.comms.send")
    session.cache.put(("k2",), 2, tool="primitives.actor.act")
    assert session.cache.invalidate(["primitives.comms"]) == 1
    assert len(session.cache) == 1
    assert session.cache.invalidate([]) == 0


# ── position tracking ──────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_runtime_tracks_iteration_and_unwinds():
    session = SteeringSession()
    seen: list = []

    class _A:
        async def b(self, _):
            seen.append(session.runtime.loop_tuple())
            return 1

    class _P:
        a = _A()

    ns = {
        CP_FN: session.cp,
        "_int": session.interrupt_point,
        AROUND_CP_FN: session.around,
        "runtime": session.runtime,
        "primitives": _P(),
    }
    await run_block("for v in [1, 2, 3]:\n    await primitives.a.b(v)\n", session, ns)
    assert seen == [(("for_1", 0),), (("for_1", 1),), (("for_1", 2),)]
    assert session.runtime.loop_tuple() == ()


@pytest.mark.asyncio
async def test_pause_holds_at_the_next_checkpoint():
    session = SteeringSession()
    ticks: list = []

    class _A:
        async def b(self, i):
            ticks.append(i)
            return i

    class _P:
        a = _A()

    ns = {
        CP_FN: session.cp,
        "_int": session.interrupt_point,
        AROUND_CP_FN: session.around,
        "runtime": session.runtime,
        "primitives": _P(),
    }
    session.runtime.pause()
    task = asyncio.create_task(
        run_block("for i in range(3):\n    await primitives.a.b(i)\n", session, ns),
    )
    for _ in range(50):
        await asyncio.sleep(0)
    assert ticks == []
    session.runtime.resume()
    await task
    assert ticks == [0, 1, 2]


# ── splicing ───────────────────────────────────────────────────────────────
def test_patch_replaces_the_named_function():
    spliced = splice_patch(
        "async def a():\n    return 1\n",
        Patch(function_name="a", source="async def a():\n    return 2\n"),
    )
    assert spliced is not None
    assert "return 2" in spliced


def test_patch_for_an_undefined_function_is_refused():
    assert (
        splice_patch(
            "async def a():\n    return 1\n",
            Patch(function_name="b", source="def b():\n    pass\n"),
        )
        is None
    )


def test_unparseable_patch_is_refused():
    assert (
        splice_patch(
            "async def a():\n    return 1\n",
            Patch(function_name="a", source="def a( :"),
        )
        is None
    )


# ── the whole mechanism, through the real sandbox ──────────────────────────
@pytest.mark.asyncio
async def test_block_without_a_session_is_untouched():
    sandbox = PythonExecutionSession()
    try:
        out = await sandbox.execute("t = 0\nfor i in range(5):\n    t += i\nt")
    finally:
        await sandbox.close()
    assert out["error"] is None
    assert out["result"] == 10


@pytest.mark.asyncio
async def test_correction_reaches_a_running_loop_and_replays_what_is_done():
    """The failure this exists for, end to end.

    Four recipients, a correction after two, and a patch that narrows the
    remainder. The already-sent recipient must replay rather than re-send, the
    one the correction excluded must never go out, and the block must finish.
    """
    sent: list[str] = []

    class _Comms:
        async def send(self, to):
            sent.append(to)
            await asyncio.sleep(0)
            return f"ok:{to}"

    class _Prims:
        comms = _Comms()

    async def author(*, interjections, session):
        assert "notify" in session.source
        return InterruptionRequest(
            reason=interjections[0],
            patches=[
                Patch(
                    function_name="notify",
                    source=(
                        "async def notify(vs):\n"
                        "    for v in vs:\n"
                        "        if v.startswith('eu-'):\n"
                        "            await primitives.comms.send(v)\n"
                        "    return 'done'\n"
                    ),
                    reason="EU only",
                ),
            ],
        )

    from unify.function_manager.steering import use_session

    interject_q: asyncio.Queue = asyncio.Queue()
    sandbox = PythonExecutionSession()
    sandbox.global_state["primitives"] = _Prims()
    steering = SteeringSession(interject_q=interject_q, patch_author=author)

    async def steer() -> None:
        while len(sent) < 2:
            await asyncio.sleep(0)
        await interject_q.put("actually only the EU ones")

    steerer = asyncio.create_task(steer())
    try:
        with use_session(steering):
            out = await sandbox.execute(
                "async def notify(vs):\n"
                "    for v in vs:\n"
                "        await primitives.comms.send(v)\n"
                "    return 'done'\n"
                "await notify(['eu-a', 'us-b', 'us-c', 'eu-d'])\n",
            )
        await steerer
    finally:
        await sandbox.close()

    assert out["error"] is None
    assert out["result"] == "done"
    assert steering.retries == 1
    # Already sent, so replayed rather than repeated.
    assert sent.count("eu-a") == 1
    # The correction arrived before this one, and prevented it.
    assert "us-c" not in sent
    # The patched code carried on from where the original stopped.
    assert "eu-d" in sent
    assert steering.cache.hits >= 1


@pytest.mark.asyncio
async def test_correction_that_cannot_be_applied_surfaces():
    """A patch naming nothing real must not spin the retry loop."""

    async def author(*, interjections, session):
        return InterruptionRequest(
            reason="nope",
            patches=[
                Patch(function_name="missing", source="def missing():\n    pass\n"),
            ],
        )

    interject_q: asyncio.Queue = asyncio.Queue()
    interject_q.put_nowait("change it")
    session = SteeringSession(interject_q=interject_q, patch_author=author)
    session.interruption = InterruptionRequest(
        reason="nope",
        patches=[Patch(function_name="notify", source="")],
    )

    async def run_source(_: str):
        raise ControlledInterruption("nope")

    with pytest.raises(RuntimeError, match="could not be applied"):
        await run_with_steering(
            "async def notify():\n    return 1\n",
            run_source,
            session=session,
        )


@pytest.mark.asyncio
async def test_stop_request_ends_without_a_retry():
    notifications: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(notification_q=notifications)
    session.interruption = InterruptionRequest(reason="no longer needed", stop=True)

    async def run_source(_: str):
        raise ControlledInterruption("no longer needed")

    with pytest.raises(ExecutionStopped, match="no longer needed") as exc_info:
        await run_with_steering(
            "value = 1\n",
            run_source,
            session=session,
        )

    assert exc_info.value.outcome == {
        "status": "stopped",
        "reason": "no longer needed",
    }
    assert session.retries == 0
    assert (await notifications.get())["type"] == "steering_stop"


@pytest.mark.asyncio
async def test_in_process_stop_is_a_clean_execution_result():
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("do not run this")
    session = SteeringSession(interject_q=queue, patch_author=_stop_author)
    sandbox = PythonExecutionSession()

    try:
        from unify.function_manager.steering import use_session

        with use_session(session):
            out = await sandbox.execute("value = 1\n")
    finally:
        await sandbox.close()

    assert out["error"] is None
    assert out["result"] == {
        "status": "stopped",
        "reason": "do not run this",
    }


@pytest.mark.asyncio
async def test_retries_are_bounded():
    session = SteeringSession()

    def _fresh_request():
        return InterruptionRequest(
            reason="again",
            patches=[
                Patch(
                    function_name="a",
                    source="async def a():\n    return 1\n",
                ),
            ],
        )

    async def run_source(_: str):
        session.interruption = _fresh_request()
        raise ControlledInterruption("again")

    session.interruption = _fresh_request()
    with pytest.raises(RuntimeError, match="gave up after"):
        await run_with_steering(
            "async def a():\n    return 0\n",
            run_source,
            session=session,
            max_retries=2,
        )
    assert session.retries == 2


# ── binding lifetime ───────────────────────────────────────────────────────
def test_binding_restores_globals_exactly():
    namespace: dict = {"unrelated": 1}
    session = SteeringSession()
    token = bind_session(namespace, session, tool_namespaces=[])
    assert current_session(namespace) is session
    restore_session(namespace, token)
    assert namespace == {"unrelated": 1}
    assert current_session(namespace) is None


def test_nested_binding_restores_the_outer_session():
    namespace: dict = {}
    outer, inner = SteeringSession(), SteeringSession()
    outer_token = bind_session(namespace, outer, tool_namespaces=[])
    inner_token = bind_session(namespace, inner, tool_namespaces=[])
    assert current_session(namespace) is inner
    restore_session(namespace, inner_token)
    assert current_session(namespace) is outer
    restore_session(namespace, outer_token)
    assert current_session(namespace) is None


def test_memoised_dispatch_does_not_stack():
    from unify.function_manager.steering import MemoisedDispatch

    class _Prims:
        pass

    session = SteeringSession()
    target = _Prims()
    twice = MemoisedDispatch(MemoisedDispatch(target, session), session)
    assert twice._target is target


@pytest.mark.asyncio
async def test_steering_reaches_a_sandbox_the_tool_never_saw():
    """The session must follow the call, not the sandbox object.

    Only one execution mode (``stateful`` with ``session_id=0``) runs in the
    sandbox that was current when the tool started. Stateless — the default for
    ``execute_code`` — builds a fresh one per call, so a session installed on
    the tool's sandbox would never be seen by the code that actually runs, and
    the common case would be silently unsteerable.
    """
    from unify.function_manager.steering import use_session

    sent: list[str] = []

    class _Comms:
        async def send(self, to):
            sent.append(to)
            await asyncio.sleep(0)
            return f"ok:{to}"

    class _Prims:
        comms = _Comms()

    async def author(*, interjections, session):
        return InterruptionRequest(
            reason=interjections[0],
            patches=[
                Patch(
                    function_name="notify",
                    source=(
                        "async def notify(vs):\n"
                        "    for v in vs:\n"
                        "        if v.startswith('eu-'):\n"
                        "            await primitives.comms.send(v)\n"
                        "    return 'done'\n"
                    ),
                ),
            ],
        )

    interject_q: asyncio.Queue = asyncio.Queue()
    steering = SteeringSession(interject_q=interject_q, patch_author=author)

    async def steer() -> None:
        while len(sent) < 2:
            await asyncio.sleep(0)
        await interject_q.put("only the EU ones")

    steerer = asyncio.create_task(steer())
    with use_session(steering):
        # Built inside the steered scope and never bound to it.
        sandbox = PythonExecutionSession()
        sandbox.global_state["primitives"] = _Prims()
        out = await sandbox.execute(
            "async def notify(vs):\n"
            "    for v in vs:\n"
            "        await primitives.comms.send(v)\n"
            "    return 'done'\n"
            "await notify(['eu-a', 'us-b', 'us-c', 'eu-d'])\n",
        )
    await steerer

    assert out["error"] is None
    assert steering.retries == 1
    assert "us-c" not in sent

    # And the probes are gone once the call is over.
    after = await sandbox.execute("t = 0\nfor i in range(4):\n    t += i\nt")
    await sandbox.close()
    assert after["result"] == 6
    assert current_session(sandbox.global_state) is None


@pytest.mark.asyncio
async def test_around_passes_through_a_synchronous_result():
    """The bracket wraps a call, not only an awaited one.

    ``user_desktop.list_linked`` is documented as synchronous, so it reaches
    ``around`` as a plain list. Awaiting that unconditionally raised "object
    list can't be used in 'await' expression", which made every synchronous
    primitive unreachable through ``execute_function``.
    """
    session = SteeringSession()

    assert await session.around("list_linked", ["a-desktop"]) == ["a-desktop"]

    async def _coro():
        return "awaited"

    assert await session.around("async call", _coro()) == "awaited"
