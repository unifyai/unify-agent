"""A pass runs under its wall-clock bound; cancellation goes through the abort path and
keeps the work as a draft; the calls are reconciled from the writer lane's journal window before the
pass is settled."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal

import pytest

from unify.memory.integration import async_pass as ap
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import _call, _writer, needs_bwrap
from tests.memory.test_writer_library_tools import EPS


class _Out:
    def __init__(self, passed):
        self.passed, self.commit = passed, "c" * 40 if passed else None


@pytest.mark.asyncio
async def test_wall_bound_cancels_through_the_abort_path():
    seen = []

    async def slow():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("abort path ran")
            raise

    sup = ap.Supervisor(0.2, asyncio.Event())
    started = time.monotonic()
    out = await sup(slow)
    assert (
        out.ended == "deadline" and out.outcome is None and seen == ["abort path ran"]
    )
    assert time.monotonic() - started < 5 and sup.last is out


@pytest.mark.asyncio
async def test_sigterm_cancels_and_a_pass_that_ends_is_returned():
    stop = asyncio.Event()
    sup = ap.Supervisor(60, stop)
    asyncio.get_running_loop().call_later(0.1, stop.set)
    assert (await sup(lambda: asyncio.sleep(3600))).ended == "cancelled"

    async def done(passed):
        return _Out(passed)

    assert (
        await ap.Supervisor(60, asyncio.Event())(lambda: done(True))
    ).ended == "landed"
    assert (
        await ap.Supervisor(60, asyncio.Event())(lambda: done(False))
    ).ended == "refused"

    async def broken():
        raise RuntimeError("x")

    assert (await ap.Supervisor(60, asyncio.Event())(broken)).ended == "error"
    assert (
        ap.Supervisor(2700, asyncio.Event()).pass_deadline_s
        == 2700 - ap.MERGE_RESERVE_S
    )


@pytest.mark.asyncio
async def test_the_window_is_the_journal_range_the_pass_ran_in(tmp_path):
    j = tmp_path / "costs.jsonl"
    j.write_text('{"origin": "request_started"}\n')
    before = j.stat().st_size

    async def writes():
        with open(j, "a") as fh:
            fh.write('{"origin": "response"}\n')
        return _Out(True)

    sup = ap.Supervisor(60, asyncio.Event(), journal=str(j), models=("m",))
    await sup(writes)
    assert sup.window == (before, j.stat().st_size)
    assert (await sup.reconcile(budget_s=0, worst_case_usd=Decimal("13.125")))[
        "calls"
    ] == 0  # no writer-lane request started


class _Blocking:
    """A model turn that writes one function and a manifest, then never answers again."""

    def __init__(self):
        self.n = 0

    async def __call__(self, messages, tools):
        self.n += 1
        if self.n > 1:
            await asyncio.Event().wait()
        code = (
            "import json, os\n"
            "os.makedirs('/memory/text', exist_ok=True)\n"
            "open('/memory/text/extra.py', 'w').write('def more(x):\\n    \"\"\"More.\"\"\"\\n    return x\\n')\n"
            "os.makedirs('/memory/.pass', exist_ok=True)\n"
            "json.dump({'items': [{'item': 'memory.text.extra:more', 'kind': 'function'}]},"
            " open('/memory/.pass/manifest.json', 'w'))\n"
        )
        return _call("c1", "execute_code", {"code": code}), "0.001"


async def _cancelled_pass(tmp_path):
    _, ev, writer = _writer(tmp_path, _Blocking())
    writer.load = EPS.__getitem__
    # the brief is not under test here (it reads the library constants; test_writer_library_brief.py pins it)
    writer._brief_library_mode = lambda: "brief"
    req = PassRequest("incremental", "svc", ["e1", "e2"], False)
    task = asyncio.ensure_future(writer.run(req, "p9"))
    for _ in range(600):
        live = getattr(writer, "_live", None)
        if live is not None and (live[0] / "text/extra.py").exists():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return ev, writer


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancelled_pass_keeps_its_work_as_a_draft(tmp_path):
    ev, writer = await _cancelled_pass(tmp_path)
    patch_blob, refused, passed = ev.db.execute(
        "SELECT patch_blob, items_refused, passed FROM passes WHERE pass_id='p9'",
    ).fetchone()
    assert patch_blob and b"text/extra.py" in writer.gate.blobs.get(patch_blob)
    assert json.loads(refused) == {"memory.text.extra:more": ["cancelled"]}
    assert passed == 0
    assert writer.mem.head() == writer.mem.head(
        "main",
    )  # nothing landed: main is where it was


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancelled_pass_round_row_holds_the_cancelled_result(tmp_path):
    """Needs pass_rounds (k) and the drafts module: runs once they are integrated."""
    drafts = pytest.importorskip("unify.memory.drafts")
    ev, writer = await _cancelled_pass(tmp_path)
    rows = ev.pass_rounds("write")
    assert rows[-1]["pass_id"] == "p9" and rows[-1]["rounds"] is None
    assert b"cancelled before the gate ran" in writer.gate.blobs.get(
        rows[-1]["gate_blob"],
    )
    assert drafts.draft_states(rows)["p9"] == "open"
