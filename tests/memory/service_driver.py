"""A stand-in writer service for the service's process tests: the real :func:`unify.memory.writer_service.service_main`
(its lock, heartbeat, stop marker, cap, lifetime, resume and termination) with a scripted pass in place of a writer
pass. :func:`unify.memory.writer_service.ensure_service` starts it through its ``argv_prefix``, exactly as it starts the
real service (the double fork, ``setsid``, the inherited lock).

The pass is a fixed number of scripted model turns through :class:`unify.memory.writer_service.ResumableTurn`; its
conversation is written to ``$DRIVER_OUT/result-<pass>.json`` and each model call to ``$DRIVER_OUT/model-calls.jsonl``.
Unless stopped, the pass is delivered its entries' messages and decides them ``nothing_to_keep``, so each entry is
handed once. ``DRIVER_BLOCK_AT=n`` makes turn n hang (once: the first time, marked by ``$DRIVER_OUT/blocked``), so a test can kill
the service mid-pass; ``DRIVER_TURN_SLEEP`` makes each turn take that long unless the service is asked to stop.
No model, network or key is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

TURNS = int(os.environ.get("DRIVER_TURNS", "5"))


async def _children_in_flight(out: Path) -> None:
    """``DRIVER_CHILDREN``: a pass with a confined box (the service's own box builder: bubblewrap with its own pid
    namespace) and a git call in flight, as a pass's cells and store writes leave them; their pids go to
    ``$DRIVER_OUT/children.json`` once the service has recorded them, then the pass waits (to be killed)."""
    import subprocess

    from unify.memory.sandbox_run import _spawn_box

    box = _spawn_box(["/bin/sh", "-c", "sleep 600"], ro=None, rw=None, cwd="/tmp", env=None, late_ro=None,
                     pass_fds=(), stdin=subprocess.DEVNULL)  # fmt: skip
    # a git call waiting on its input, in the service's group (hash-object needs no repository)
    git = subprocess.Popen(["git", "hash-object", "--stdin"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                           cwd="/tmp")  # fmt: skip
    inside = None
    for _ in range(200):  # the box's own process (its pid namespace's), once bubblewrap has started it
        kids = [int(c) for t in os.listdir(f"/proc/{box.pid}/task")
                for c in open(f"/proc/{box.pid}/task/{t}/children").read().split()]  # fmt: skip
        if kids:
            inside = kids[0]
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(2.5)  # past MEMBERS_S: the service has recorded them
    (out / "children.json").write_text(json.dumps({
        "service": os.getpid(), "box": box.pid, "inside": inside, "git": git.pid,
        "box_ns": os.readlink(f"/proc/{inside}/ns/pid") if inside else None,
    }))  # fmt: skip
    await asyncio.sleep(3600)


async def _http_call(out: Path, pass_id: str, url: str) -> None:
    """``DRIVER_HTTP``: one model call of the pass, a POST to a stand-in endpoint that answers slowly; its outcome to
    ``$DRIVER_OUT/http-<pass>.json``."""
    import urllib.request

    def call() -> dict:
        req = urllib.request.Request(url, data=b'{"messages": []}', headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return {"ok": True, "status": resp.status, "body": resp.read().decode()}
        except Exception as exc:  # noqa: BLE001 - recorded
            return {"ok": False, "error": type(exc).__name__}

    got = await asyncio.get_running_loop().run_in_executor(None, call)
    (out / f"http-{pass_id}.json").write_text(json.dumps(got))


def _box_read(out: Path, paths: list[str]) -> None:
    """``DRIVER_BOX_READ``: a cell in the pass's own confined box tries to open each path (the run record's files) for
    reading and appending; what it saw to ``$DRIVER_OUT/box-read.txt``."""
    from unify.memory.sandbox_run import run_confined

    code = (
        "import sys\n"
        "for p in sys.argv[1:]:\n"
        "    for mode in ('r', 'a'):\n"
        "        try:\n"
        "            open(p, mode).close()\n"
        "            print('OPENED', mode, p)\n"
        "        except OSError as e:\n"
        "            print('REFUSED', mode, type(e).__name__)\n"
    )
    res = run_confined(["/usr/bin/python3", "-I", "-c", code, *paths], timeout_s=60)
    (out / "box-read.txt").write_text(res.stdout)


async def runner(
    home, pass_id, handed, *, resume, folder, journal_start, effort, settings, stop=None
):
    from unify.memory import attention
    from unify.memory.integration.consolidate import AttentionPassReport
    from unify.memory.writer_service import ResumableTurn

    out = Path(os.environ["DRIVER_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("DRIVER_CHILDREN"):
        await _children_in_flight(out)
    if os.environ.get("DRIVER_HTTP"):
        await _http_call(out, pass_id, os.environ["DRIVER_HTTP"])
    if os.environ.get("DRIVER_BOX_READ"):
        _box_read(out, os.environ["DRIVER_BOX_READ"].split(":"))
    if os.environ.get("DRIVER_TEMP"):
        # a pass's temporary folder, as a writer pass makes one
        import tempfile

        made = tempfile.mkdtemp(prefix="pass-")
        (out / f"temp-{pass_id}.txt").write_text(made)
    block_at = int(os.environ.get("DRIVER_BLOCK_AT", "-1"))
    turn_sleep = float(os.environ.get("DRIVER_TURN_SLEEP", "0"))

    async def scripted(messages, tools):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        with open(out / "model-calls.jsonl", "a") as fh:
            fh.write(
                json.dumps(
                    {"pass": pass_id, "turn": n, "pid": os.getpid(), "resume": resume}
                )
                + "\n"
            )
        if n == block_at and not (out / "blocked").exists():
            (out / "blocked").write_text(str(os.getpid()))
            await asyncio.sleep(3600)
        if turn_sleep:
            try:
                (
                    await asyncio.wait_for(stop.wait(), turn_sleep)
                    if stop is not None
                    else await asyncio.sleep(turn_sleep)
                )
            except asyncio.TimeoutError:
                pass
        return {
            "role": "assistant",
            "content": f"reply {n} to {messages[-1]['content']}",
        }, "0.001"

    turn = ResumableTurn(scripted, folder, inputs={"pass_id": pass_id})
    messages = [{"role": "user", "content": "go"}]
    for _ in range(TURNS):
        if stop is not None and stop.is_set():
            break
        msg, _usd = await turn(messages, [])
        messages.append(msg)
        messages.append({"role": "user", "content": f"next {len(messages)}"})
    (out / f"result-{pass_id}.json").write_text(json.dumps(messages))
    stopped = stop is not None and stop.is_set()
    seqs = [e.refs.record_seq for e in handed if e.refs.record_seq is not None]
    if not stopped and handed:
        # the writer was delivered its entries' messages and decides them, as a pass that read them does
        attention.Ledger(home).decide(
            pass_id,
            [e.entry for e in handed],
            "nothing_to_keep",
            reason="the scripted pass keeps nothing",
            reads=attention.PassReads(),
            delivered=seqs,
            decisions_path=Path(folder) / "decisions.jsonl",
        )
    return AttentionPassReport(
        pass_id,
        [e.entry for e in handed],
        "cancelled" if stop is not None and stop.is_set() else "landed",
        killed_turns=list(turn.killed),
        turns=turn.n,
        replayed_turns=turn.replayed,
        reads=attention.PassReads(),
        delivered=[] if stopped else seqs,
        usd_known="0.005",
        settle_usd="0.005",
    )


def main(argv=None) -> int:
    from unify.memory import writer_service as ws
    from unify.memory.integration import request

    args = ws._parse(argv)
    ws.release_route_fds()  # as ws.main does, before the fork
    out = os.environ.get("DRIVER_OUT")
    if out:
        from unify.memory.integration import switch

        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"routes-{os.getpid()}.json").write_text(
            json.dumps(
                {
                    "writer": bool(switch._FD_TOKEN is not None and switch._FD_TOKEN.get_secret_value()),
                    "helper": switch.helper_lane_route()[0] is not None,
                }
            )
        )
    if os.environ.get("DRIVER_OUT") and os.environ.get("DRIVER_LIST_HELD"):
        # what the model client holds: any provider key it read would be a non-empty secret setting
        import unillm
        from pydantic import SecretStr

        held = [n for n in type(unillm.SETTINGS).model_fields
                if isinstance(getattr(unillm.SETTINGS, n, None), SecretStr)
                and getattr(unillm.SETTINGS, n).get_secret_value()]  # fmt: skip
        (Path(os.environ["DRIVER_OUT"]) / "keys.json").write_text(json.dumps({"held": held}))
    if args.daemon:
        ws.daemonize()
    if os.environ.get("DRIVER_PASS_WALL_S"):  # a pass's wall clock, short for the test
        from unify.memory.integration import consolidate, switch

        switch.PASS_WALL_S = consolidate.PASS_WALL_S = float(os.environ["DRIVER_PASS_WALL_S"])
    ws.check_writer_settings = lambda settings: None
    request.actor_effort = lambda: "low"
    asyncio.run(
        ws.service_main(
            args.home,
            runner=runner,
            poll_s=float(os.environ.get("DRIVER_POLL_S", "0.2")),
            idle_exit_s=float(os.environ.get("DRIVER_IDLE_S", "600")),
            heartbeat_s=float(os.environ.get("DRIVER_BEAT_S", "30")),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
