"""Whether a real model writes a usable correction for running code.

The machinery is covered symbolically in ``test_live_steering``. What is not
covered there is the judgement the mechanism now depends on: given a running
block, a correction in plain English, and a list of what has already happened,
does the model produce a patch that changes the right thing and repeats
nothing?

Real LLM calls, cached per unique input like everywhere else in the suite.
Assertions are on the *outcome* — which sends happened — rather than on the
text of the patch, because there are many correct rewrites and only one
correct set of side effects.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from unify.actor.code_act_actor import _synthesize_python_call
from unify.actor.execution.session import PythonExecutionSession
from unify.function_manager.steering import SteeringSession, use_session
from unify.function_manager.steering_patcher import build_patch_author
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (steering)
pytestmark = [pytest.mark.eval, pytest.mark.usefixtures("as_shipped")]


class _Comms:
    """Records who was contacted, and yields so a correction can land."""

    def __init__(self) -> None:
        self.sent: List[str] = []

    async def send(self, to: str) -> str:
        self.sent.append(to)
        await asyncio.sleep(0)
        return f"sent:{to}"


class _Prims:
    def __init__(self, comms: _Comms) -> None:
        self.comms = comms


async def _steer_after(comms: _Comms, count: int, queue: asyncio.Queue, text: str):
    """Deliver *text* once *count* sends have landed.

    Triggered on observed state rather than a sleep, so the correction arrives
    at the same point whether the LLM call is cached (milliseconds) or live.
    """

    async def _run() -> None:
        while len(comms.sent) < count:
            await asyncio.sleep(0)
        await queue.put(text)

    return asyncio.create_task(_run())


async def _run_steered(source: str, comms: _Comms, correction: str, *, after: int):
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=build_patch_author())
    sandbox = PythonExecutionSession()
    sandbox.global_state["primitives"] = _Prims(comms)

    steerer = await _steer_after(comms, after, queue, correction)
    try:
        with use_session(session):
            out = await sandbox.execute(source)
        await steerer
    finally:
        await sandbox.close()
    return out, session


# ── execute_code ───────────────────────────────────────────────────────────
EU_BLOCK = (
    "async def notify(vendors):\n"
    "    for v in vendors:\n"
    "        await primitives.comms.send(v)\n"
    "    return 'notified'\n"
    "await notify(['eu-alpha', 'us-beta', 'us-gamma', 'eu-delta'])\n"
)


@pytest.mark.asyncio
async def test_model_narrows_a_running_loop_without_repeating_sends():
    """The scenario the whole mechanism exists for.

    Four recipients, a correction after two, and no instruction about what to
    do with the ones already contacted. A usable patch stops the excluded
    recipients and does not re-contact anybody.
    """
    comms = _Comms()
    out, session = await _run_steered(
        EU_BLOCK,
        comms,
        "Hold on — only the EU vendors should get this, not the US ones.",
        after=2,
    )

    assert out["error"] is None, out["error"]
    assert session.retries >= 1, "the correction never reached the running loop"

    # The correction's whole point: the excluded recipient must not be sent to.
    assert "us-gamma" not in comms.sent
    # Nobody contacted twice, even though the block re-ran from the top.
    assert len(comms.sent) == len(set(comms.sent)), comms.sent
    # The already-contacted EU vendor stays contacted, and the remaining one
    # is picked up by the patched code.
    assert "eu-alpha" in comms.sent
    assert "eu-delta" in comms.sent


@pytest.mark.asyncio
async def test_model_leaves_completed_work_alone_when_told_to_stop():
    """ "Stop" cannot un-send. A usable patch does not pretend otherwise."""
    comms = _Comms()
    out, session = await _run_steered(
        EU_BLOCK,
        comms,
        "Stop — do not contact anyone else.",
        after=2,
    )

    assert out["error"] is None, out["error"]
    already = set(comms.sent[:2])
    # Whatever the patch decided, nothing new went out beyond what had already
    # been sent when the correction arrived.
    assert set(comms.sent) == already, comms.sent


# ── execute_function ───────────────────────────────────────────────────────
STORED_IMPLEMENTATION = (
    "async def notify_vendors(vendors):\n"
    "    results = []\n"
    "    for v in vendors:\n"
    "        results.append(await primitives.comms.send(v))\n"
    "    return results\n"
)


class _StubFunctionStore:
    """Just enough of a FunctionManager for the synthesiser to emit a preamble.

    The LLM is never stubbed; this only stands in for row storage so the test
    does not need a live backend to reach the code path under test.
    """

    def __init__(self, implementation: str) -> None:
        self.implementation = implementation
        self.writes: List[Any] = []

    def _get_function_data_by_name(self, name: str) -> dict:
        return {
            "name": name,
            "implementation": self.implementation,
            "is_primitive": False,
        }


@pytest.mark.asyncio
async def test_steering_execute_function_diverges_from_the_stored_source():
    """A steered call may stop resembling the function it started as.

    ``execute_function`` synthesises the stored implementation as a preamble,
    so a correction rewrites that preamble for this run. Deviating from the
    stored lines is expected — the row is not the thing executing. Persisting
    anything learned here is StorageCheck's decision afterwards, not a side
    effect of steering.
    """
    store = _StubFunctionStore(STORED_IMPLEMENTATION)
    source = _synthesize_python_call(
        function_name="notify_vendors",
        call_kwargs={"vendors": ["eu-alpha", "us-beta", "us-gamma", "eu-delta"]},
        function_manager=store,
    )
    assert "async def notify_vendors" in source, "expected a preamble to steer"

    comms = _Comms()
    out, session = await _run_steered(
        source,
        comms,
        "Only the EU vendors, please — skip the US ones.",
        after=2,
    )

    assert out["error"] is None, out["error"]
    assert session.retries >= 1
    assert "us-gamma" not in comms.sent
    assert len(comms.sent) == len(set(comms.sent)), comms.sent

    # The stored definition is untouched: steering edits the running copy.
    assert store.implementation == STORED_IMPLEMENTATION
    assert store.writes == []
    # And the source that actually ran is no longer the stored source.
    assert session.source != source


@pytest.mark.asyncio
async def test_patch_author_declines_when_nothing_needs_changing():
    """A remark that changes no remaining work should not force a rewrite.

    Retrying costs a full re-execution, so a correction that does not alter
    what is left to do must leave the block running as written.
    """
    comms = _Comms()
    out, session = await _run_steered(
        EU_BLOCK,
        comms,
        "Thanks, this all looks right — carry on.",
        after=2,
    )

    assert out["error"] is None, out["error"]
    assert session.retries == 0, "an approving remark should not trigger a rewrite"
    assert comms.sent == ["eu-alpha", "us-beta", "us-gamma", "eu-delta"]


# ── stop judgment on unpatchable source ──────────────────────────────────────
# A block that defines no function cannot be rewritten mid-flight, so the only
# call the author can make there is stop-or-continue. That judgment is the
# entire product surface of steering such a block: too eager and benign
# chatter kills real work, too timid and "stop" gets ignored.

DEPLOY_SCRIPT = (
    "import subprocess\n"
    "subprocess.run(['aws', 's3', 'sync', './build', 's3://prod-assets', '--delete'], check=True)\n"
    "subprocess.run(['aws', 'cloudfront', 'create-invalidation', "
    "'--distribution-id', 'E2EXAMPLE', '--paths', '/*'], check=True)\n"
)


async def _stop_decision(correction: str):
    session = SteeringSession()
    session.bind_source(DEPLOY_SCRIPT)
    author = build_patch_author()
    return await author(interjections=[correction], session=session)


@pytest.mark.asyncio
async def test_model_stops_unpatchable_work_when_the_task_is_revoked():
    request = await _stop_decision(
        "Wait, stop — that's deploying to prod and we're not ready.",
    )
    assert request is not None, "a revoking correction was ignored"
    assert request.stop is True
    assert request.patches == []


@pytest.mark.asyncio
async def test_model_does_not_stop_for_a_question():
    request = await _stop_decision("How long does this usually take?")
    assert (
        request is None or request.stop is False
    ), "a question about the run must not kill it"


@pytest.mark.asyncio
async def test_model_does_not_stop_for_encouragement():
    request = await _stop_decision("Great, looks right — keep going.")
    assert (
        request is None or request.stop is False
    ), "an approving remark must not kill the run"
