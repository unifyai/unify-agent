"""Writer-launched helpers (the ``delegate`` tool): staging format, readability by the writer, concurrency, spend,
events, failures and the helper box's mounts. Keyless: every model is a scripted fake.
"""

import asyncio
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory_v2 import helpers, tool_spec, views
from unify.memory_v2.episodes import Cell
from unify.memory_v2.integration import consolidate, switch
from unify.memory_v2.sandbox_run import SandboxResult
from tests.memory_v2.test_writer_library_tools import _episode
from tests.memory_v2.test_writer_pass import needs_bwrap

EPS = {
    "e1": _episode(
        "e1",
        [Cell(0, "def area(w, h):\n    return w * h\nprint(area(2, 3))", "6\n")],
    ),
    "e2": _episode("e2", [Cell(0, "print('two')", "two\n")]),
    "e3": _episode("e3"),  # recorded, not named by any task
}
CANDIDATE = 'def area(w, h):\n    """Area of a w by h rectangle."""\n    return w * h\n'
CASE = json.dumps([{"args": [2, 3], "result": 6}])


def _tc(cid, name, args):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _msg(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class Script:
    """A fresh scripted helper model per delegate call, chosen by the task text in the first user message."""

    def __init__(self, scripts, usd="0.001", gap=0.0):
        self.scripts, self.usd, self.gap = scripts, usd, gap
        self.order: list[str] = []
        self.outputs: dict[str, str] = {}
        self.in_flight = self.max_in_flight = 0
        self.prefixes: list[str] = []

    def factory(self, session=None):
        state = {"queue": None}

        async def turn(messages, tools):
            key = messages[1]["content"].split("\n")[1]
            if state["queue"] is None:
                state["queue"] = list(self.scripts[key])
                self.prefixes.append(json.dumps(messages[0], sort_keys=True))
            for m in messages:
                if m.get("role") == "tool":
                    self.outputs[m["tool_call_id"]] = m["content"]
            self.order.append(key)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await asyncio.sleep(self.gap)
            finally:
                self.in_flight -= 1
            q = state["queue"]
            if isinstance(q[0], Exception):
                raise q.pop(0)
            return q.pop(0), self.usd

        return turn


def _ctx(tmp_path, script=None, **kw):
    inputs, lib = tmp_path / "inputs", tmp_path / "memory"
    inputs.mkdir(parents=True)
    (inputs / "pool").mkdir()  # the writer's own inputs: never shown to a helper
    (inputs / "pool" / "INDEX.md").write_text("pool")
    lib.mkdir()
    (lib / "INDEX.md").write_text("# library\n")
    (lib / ".pass").mkdir()
    (lib / ".pass" / "manifest.json").write_text("{}")
    base = dict(
        pass_id="p1",
        out_root=tmp_path / "state" / helpers.HELPERS_DIR / "p1",
        inputs=inputs,
        library=lib,
        load=EPS.__getitem__,
        recorded=lambda: list(EPS),
        new_turn=script.factory if script is not None else None,
        deadline=time.monotonic() + 600,
        spend=SimpleNamespace(usd=Decimal("0"), unknown=0),
        scratch=tmp_path / "scratch",
        events=tmp_path / "out" / "tool-events.jsonl",
    )
    base.update(kw)
    return helpers.DelegateContext(**base)


def _drafting(task_key="draft area"):
    return {
        task_key: [
            _msg(
                _tc("o", "overview", {"episode": "e1"}),
                _tc("r", "read", {"path": "/inputs/episodes/e1.json"}),
            ),
            _msg(_tc("x", "execute_code", {"code": "print(6)"})),
            _msg(
                _tc("c", "stage", {"path": "candidates/area.py", "text": CANDIDATE}),
                _tc("k", "stage", {"path": "cases/area.json", "text": CASE}),
                _tc("bad", "stage", {"path": "../escape.py", "text": "x"}),
            ),
            _msg(_tc("d", "done", {"summary": "drafted area from e1"})),
        ],
    }


@pytest.fixture
def fake_box(monkeypatch):
    seen = []

    def run(argv, **kw):
        listing = {
            dst: sorted(str(p.relative_to(src)) for p in Path(src).rglob("*"))
            for src, dst in kw["ro"].items()
        }
        seen.append({**kw, "argv": argv, "listing": listing})
        return SandboxResult(0, "6\n", "", False)

    monkeypatch.setattr(helpers, "run_confined", run)
    return seen


def test_delegate_stages_in_the_fork_format_and_the_writer_reads_it(tmp_path, fake_box):
    script = Script(_drafting())
    ctx = _ctx(tmp_path, script)
    reply = asyncio.run(helpers.handle({"task": "draft area", "episodes": ["e1"]}, ctx))
    assert reply.startswith("helper call-0: ok")
    assert (
        "candidates/area.py" in reply
        and "cases/area.json" in reply
        and "drafted area from e1" in reply
    )
    out = ctx.out_root / "call-0"
    assert (out / "candidates" / "area.py").read_text() == CANDIDATE
    assert (out / "cases" / "area.json").read_text() == CASE
    status = json.loads((out / "helper.json").read_text())
    assert (
        status["status"] == "ok"
        and status["usd"] == "0.004"
        and status["unknown_cost_calls"] == 0
    )
    assert [f["path"] for f in status["files"]] == [
        "candidates/area.py",
        "cases/area.json",
    ]
    assert status["task_sha256"] and status["episodes"] == ["e1"]
    # the writer reads the helper's files with its own reader, under /inputs/helpers/<call>/
    roots = {"/inputs": ctx.inputs}
    assert "def area" in views.read("/inputs/helpers/call-0/candidates/area.py", roots)
    assert '"status": "ok"' in views.read("/inputs/helpers/call-0/helper.json", roots)
    assert "not a plain relative path" in script.outputs["bad"]
    assert "6" in script.outputs["x"] and script.outputs["o"].strip()
    # spend joins the pass's and is recorded per call
    assert ctx.spend.usd == Decimal("0.004") and ctx.spend.unknown == 0
    assert ctx.records[0]["usd"] == "0.004" and len(ctx.records[0]["files"]) == 2


def test_the_helper_box_is_ro_ro_rw_and_holds_only_what_the_task_needs(
    tmp_path,
    fake_box,
):
    ctx = _ctx(tmp_path, Script(_drafting()), shared={"b1": "shared block"})
    asyncio.run(helpers.handle({"task": "draft area", "episodes": ["e1"]}, ctx))
    (call,) = fake_box
    assert sorted(call["ro"].values()) == ["/cell", "/inputs", "/memory"]
    assert list(call["rw"].values()) == ["/work"] and call["cwd"] == "/work"
    # the open log's pipe is the one extra descriptor and variable; the interpreter skips the user site
    (fd,) = call["pass_fds"]
    assert call["env"] == {"PYTHONPATH": "/inputs:/memory", "MEMORY_BOX_OPEN_LOG": f"fd:{fd}"}
    assert call["argv"][-2:] == ["-s", "/cell/code.py"]
    assert (
        f"--fsize={helpers.WORK_FILE_BYTES}:{helpers.WORK_FILE_BYTES}" in call["argv"]
    )
    # /inputs: the named episode's record, the shared blocks and the open-log hook; no pool, gate, drafts or other
    # helpers' output
    assert call["listing"]["/inputs"] == [
        "episodes",
        "episodes/e1.json",
        "shared",
        "shared/b1.txt",
        "sitecustomize.py",
    ]
    # /memory: the library as the writer sees it, without the writer's .pass
    assert call["listing"]["/memory"] == ["INDEX.md"]


def test_delegate_calls_in_one_turn_run_concurrently_with_serialised_model_calls(
    tmp_path,
    fake_box,
):
    def two(key):
        return [
            _msg(
                _tc(f"{key}s", "stage", {"path": "notes.md", "text": f"lesson {key}"}),
            ),
            _msg(_tc(f"{key}d", "done", {"summary": key})),
        ]

    script = Script({"A": two("A"), "B": two("B")}, gap=0.01)
    ctx = _ctx(tmp_path, script)
    replies = asyncio.run(
        helpers.handle_turn(
            [{"task": "A", "episodes": ["e1"]}, {"task": "B", "episodes": ["e2"]}],
            ctx,
        ),
    )
    assert replies[0].startswith("helper call-0: ok") and replies[1].startswith(
        "helper call-1: ok",
    )
    assert script.order == ["A", "B", "A", "B"]  # both helpers were live at once
    assert script.max_in_flight == 1  # one helper model call in flight at a time
    assert len(set(script.prefixes)) == 1  # siblings share the brief byte for byte
    assert (ctx.out_root / "call-1" / "notes.md").read_text() == "lesson B"
    assert ctx.spend.usd == Decimal("0.004") and len(ctx.records) == 2


def test_every_helper_tool_call_is_an_event(tmp_path, fake_box):
    ctx = _ctx(tmp_path, Script(_drafting()))
    asyncio.run(helpers.handle({"task": "draft area", "episodes": ["e1"]}, ctx))
    rows = [json.loads(x) for x in ctx.events.read_text().splitlines()]
    assert [r["tool"] for r in rows] == [
        "overview",
        "read",
        "execute_code",
        "stage",
        "stage",
        "stage",
        "done",
    ]
    assert all(
        r["agent"] == "helper" and r["pass_id"] == "p1" and r["helper_call"] == "call-0"
        for r in rows
    )
    assert [r["call"] for r in rows] == list(range(7))
    # usage on the first call of each model turn only
    assert [r["usage"] is not None for r in rows] == [
        True,
        False,
        True,
        True,
        False,
        False,
        True,
    ]
    assert rows[0]["usage"]["usd"] == "0.001"
    bad = rows[5]
    assert (
        bad["status"] == "refused"
        and bad["refusal"]["form"] == "path"
        and bad["paths"] == []
    )
    assert rows[3]["paths"] == [
        {"path": "/inputs/helpers/call-0/candidates/area.py", "mode": "write"},
    ]
    assert rows[1]["paths"] == [{"path": "/inputs/episodes/e1.json", "mode": "read"}]


def test_a_failing_helper_returns_a_clear_result_and_never_fails_the_pass(
    tmp_path,
    fake_box,
):
    script = Script({"boom": [RuntimeError("upstream said no")]})
    ctx = _ctx(tmp_path, script)
    reply = asyncio.run(helpers.handle({"task": "boom", "episodes": ["e1"]}, ctx))
    assert (
        reply.startswith("helper call-0: error: RuntimeError")
        and "upstream said no" not in reply
    )
    assert (
        ctx.spend.unknown == 1
    )  # a failed call may have cost money: booked as unknown
    status = json.loads((ctx.out_root / "call-0" / "helper.json").read_text())
    assert status["status"] == "error: RuntimeError" and status["usd"] == "unknown"
    assert (ctx.inputs / "helpers" / "call-0" / "helper.json").is_file()


def test_unpriced_helper_calls_stay_unknown(tmp_path, fake_box):
    script = Script(
        {"t": [_msg(_tc("d", "done", {"summary": "nothing"}))]},
        usd="unknown",
    )
    ctx = _ctx(tmp_path, script)
    reply = asyncio.run(helpers.handle({"task": "t", "episodes": []}, ctx))
    assert "spend: unknown USD (1 unpriced calls" in reply and ctx.spend.unknown == 1


def test_refusals_start_no_helper(tmp_path):
    ctx = _ctx(tmp_path, None, lane_refusal="no helper lane")
    assert asyncio.run(helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)) == (
        "refused: no helper can start: no helper lane"
    )
    ctx = _ctx(tmp_path / "b", Script({}))
    for args, phrase in (
        ({"task": "", "episodes": ["e1"]}, "task must be"),
        ({"task": "t", "episodes": "e1"}, "episodes must be"),
        ({"task": "t", "episodes": ["nope"]}, "'nope' is not a recorded episode"),
        ({"task": "t", "episodes": ["../e1"]}, "is not a recorded episode"),
    ):
        assert phrase in asyncio.run(helpers.handle(args, ctx))
    late = _ctx(tmp_path / "c", Script({}), deadline=time.monotonic())
    assert "wall clock is used up" in asyncio.run(
        helpers.handle({"task": "t", "episodes": []}, late),
    )
    assert not (tmp_path / "b" / "state").exists()


def test_helpers_read_any_recorded_episode_but_only_through_the_index(
    tmp_path,
    fake_box,
):
    script = Script(
        {
            "t": [
                _msg(
                    _tc("other", "read_episode", {"episode": "e3"}),
                    _tc("hidden", "read_episode", {"episode": "e4"}),
                    _tc("pathy", "overview", {"episode": "../../etc/passwd"}),
                    _tc("pool", "read", {"path": "/inputs/pool/INDEX.md"}),
                ),
                _msg(_tc("d", "done", {"summary": "read"})),
            ],
        },
    )
    loads = []

    def load(eid):
        loads.append(eid)
        return {**EPS, "e4": _episode("e4")}[eid]

    ctx = _ctx(tmp_path, script, load=load)
    asyncio.run(helpers.handle({"task": "t", "episodes": ["e1"]}, ctx))
    assert "request" in script.outputs["other"]  # e3 was not named, and is readable
    assert (
        "not a recorded episode" in script.outputs["hidden"]
    )  # loadable, but not in the index
    assert "not an episode id" in script.outputs["pathy"]
    assert "e4" not in loads and all(".." not in e for e in loads)
    assert script.outputs["pool"].startswith(
        "refused",
    )  # the writer's pool is not in a helper's /inputs


def test_staging_again_replaces_and_the_quota_is_stated(tmp_path, monkeypatch):
    st = helpers._Staging(tmp_path / "s" / "call-0")
    assert st.stage("notes.md", "one") == "staged notes.md (3 bytes)"
    assert st.stage("notes.md", "two!") == "staged notes.md (4 bytes)"
    assert (tmp_path / "s" / "call-0" / "notes.md").read_text() == "two!"
    monkeypatch.setattr(helpers, "QUOTA_FILES", 1)
    assert "staging quota" in st.stage("candidates/f.py", "def f(): pass")
    assert "outside candidates" in st.stage("evil.sh", "x")
    with pytest.raises(FileExistsError):
        helpers._Staging(tmp_path / "s" / "call-0")  # always a fresh directory


def test_delegate_is_a_write_tool_offered_only_with_a_helper_lane():
    write = [s.name for s in tool_spec.tools_for("write", frozenset({"helper"}))]
    assert "delegate" in write
    assert "delegate" not in [s.name for s in tool_spec.tools_for("write")]
    assert "delegate" not in [
        s.name for s in tool_spec.tools_for("curate", frozenset({"helper"}))
    ]
    names = [t["function"]["name"] for t in helpers.helper_tools()]
    assert "delegate" not in names and {"stage", "done", "execute_code"} <= set(names)


def test_the_analysts_setting_is_retired_loudly():
    assert (
        switch.parse_library_analysts("") == "off"
        and switch.parse_library_analysts("off") == "off"
    )
    with pytest.raises(ValueError, match="retired"):
        switch.parse_library_analysts("writer")


def test_helper_model_defaults_to_the_forks_model():
    s = SimpleNamespace(
        UNIFY_MEMORY_LIBRARY_HELPER_MODEL="",
        UNIFY_MODEL="vendor/actor-model@openrouter",
    )
    assert switch.library_helper_model(s) == "vendor/actor-model@openrouter"
    s.UNIFY_MEMORY_LIBRARY_HELPER_MODEL = "vendor/other"
    assert switch.library_helper_model(s) == "vendor/other"
    with pytest.raises(ValueError):
        switch.parse_library_helper_model("two words")


def test_the_helper_lane_reads_its_token_from_a_descriptor_and_scrubs_the_environment(
    monkeypatch,
):
    for name, value in (
        ("_HELPER_TOKEN", None),
        ("_HELPER_BASE", ""),
        ("_HELPER_TRIED", False),
        ("_HELPER_REFUSAL", None),
    ):
        monkeypatch.setattr(switch, name, value)
    assert switch.helper_lane_route() == (None, "no helper lane")
    r, w = os.pipe()
    os.write(w, b"helper-token-0123456789abcdef\n")  # pragma: allowlist secret
    os.close(w)
    os.set_inheritable(r, True)
    base = "http://127.0.0.1:18081/helper/v1"
    env = {switch.LIBRARY_HELPER_BASE_URL: base, switch.LIBRARY_HELPER_TOKEN_FD: str(r)}
    s = SimpleNamespace(**env)
    assert switch.settle_helper_route_env(env, s) is None
    assert env == {}  # both names left the environment
    pair, why = switch.helper_lane_route()
    assert (
        why is None
        and pair[0] == base
        and pair[1].get_secret_value()
        == "helper-token-0123456789abcdef"  # pragma: allowlist secret
    )
    with pytest.raises(OSError):
        os.fstat(r)  # the descriptor was closed once read
    with pytest.raises(ValueError):
        switch.parse_library_helper_base_url("http://127.0.0.1:18081/fork/v1")


def test_no_helper_lane_means_no_delegate_and_no_fallback(monkeypatch):
    for name, value in (
        ("_HELPER_TOKEN", None),
        ("_HELPER_BASE", ""),
        ("_HELPER_TRIED", True),
        ("_HELPER_REFUSAL", None),
    ):
        monkeypatch.setattr(switch, name, value)
    s = SimpleNamespace(
        UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL="",
        UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD="",
        UNIFY_MEMORY_LIBRARY_HELPER_MODEL="",
        UNIFY_MODEL="vendor/actor-model@openrouter",
    )
    got = consolidate._helper_turns(s, "low", True)
    assert got == {"helper_turn": None, "helper_refusal": "no helper lane"}
    assert consolidate._helper_turns(s, "low", False) == {}


@needs_bwrap
def test_the_real_helper_box_writes_only_work(tmp_path):
    code = (
        "import pathlib\n"
        "pathlib.Path('/work/x.txt').write_text('ok')\n"
        "for p in ('/memory/y.txt', '/inputs/z.txt'):\n"
        "    try:\n"
        "        pathlib.Path(p).write_text('no')\n"
        "        print('WROTE', p)\n"
        "    except OSError as e:\n"
        "        print('DENIED', p)\n"
        "print(open('/memory/INDEX.md').read().strip())\n"
    )
    script = Script(
        {
            "t": [
                _msg(_tc("x", "execute_code", {"code": code})),
                _msg(_tc("d", "done", {"summary": "s"})),
            ],
        },
    )
    ctx = _ctx(tmp_path, script)
    asyncio.run(helpers.handle({"task": "t", "episodes": ["e1"]}, ctx))
    out = script.outputs["x"]
    assert (
        "DENIED /memory/y.txt" in out
        and "DENIED /inputs/z.txt" in out
        and "WROTE" not in out
    )
    assert "# library" in out
    assert not (ctx.library / "y.txt").exists() and not (ctx.inputs / "z.txt").exists()
