"""The writer's tool surface: tool JSON rendered from the one spec through the agent profiles, the pool tool,
structured tool events and the removed reading rule (rev 2: no curator; every call is a cell)."""

import asyncio
import json


from unify.memory import agent_profiles, pool, tool_spec
from unify.memory import writer_tools as wt
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import EPS

def test_every_hand_off_has_a_producer_and_a_consumer():
    rows = agent_profiles.AGENT_PROFILES
    agents = {r["agent"] for r in rows}
    produced = {(p["artifact"], p["consumer"]) for r in rows for p in r["produces"]}
    consumed = {(c["artifact"], r["agent"]) for r in rows for c in r["consumes"]}
    assert consumed == produced
    assert {c for _, c in produced} <= agents


def test_the_call_record_is_a_gate_internal_hand_off():
    """The gate's test runs write the call record in its box; the gate consumes it, the tests the writer wrote produce
    it (rev 2 has no curator), and no model-facing agent is handed it."""
    from unify.memory import example_calls

    rows = {r["agent"]: r for r in agent_profiles.AGENT_PROFILES}
    record = {"artifact": "call record", "path": example_calls.CALLS_OUT}
    assert record in rows["gate"]["consumes"]
    assert "curator" not in rows
    for agent in ("writer",):
        assert {**record, "consumer": "gate"} in rows[agent]["produces"], agent
    for agent, row in rows.items():
        if agent != "gate":
            assert all(c["artifact"] != "call record" for c in row["consumes"]), agent
        assert all(
            p.get("consumer") == "gate"
            for p in row["produces"]
            if p["artifact"] == "call record"
        ), agent
    mounts = {m["path"]: m["mode"] for m in rows["gate"]["mounts"]}
    assert mounts[example_calls.CALLS_DIR] == "ro"
    assert mounts[example_calls.CALLS_OUT.rsplit("/", 1)[0]] == "rw"
    # the writer's own box is unchanged: it never sees the record
    for agent in ("writer",):
        assert not any(
            m["path"] in (example_calls.CALLS_DIR, "/calls")
            for m in rows[agent]["mounts"]
        ), agent


def test_every_refusal_form_renders_and_every_described_parameter_is_in_the_schema():
    for spec in tool_spec.registry():
        props = spec.parameters.get("properties", {})
        for req in spec.parameters.get("required", []):
            assert req in props, (spec.name, req)
        for form in spec.refusals:
            assert form.isidentifier()
    assert tool_spec.refusal("pool", "bad_op").startswith("refused: ")
    assert wt.general("unknown_tool", name="x", tools="a, b").text == (
        "refused: unknown tool 'x'; your tools: a, b"
    )


# --- the pool tool --------------------------------------------------------------------------------------------


def _staged_pool(tmp_path):
    p = pool.Pool(None)
    a = p.upsert(pool.deferred("e1", "a grid rule"), "p1")
    b = p.upsert(pool.deferred("e2", "a parser that may recur"), "p1")
    p.stage(tmp_path / "pool")
    (tmp_path / "box").mkdir()
    return a.id, b.id, tmp_path / "pool", tmp_path / "box"


def test_pool_list_filter_show(tmp_path):
    a, b, pdir, box = _staged_pool(tmp_path)
    got = wt.pool({"op": "list"}, pdir, box)
    assert got.status == "ok" and got.text.startswith("2 open entries")
    assert f"`{a}` deferred" in got.text and f"`{b}` deferred" in got.text
    got = wt.pool({"op": "filter", "text": "PARSER"}, pdir, box)
    assert b in got.text and a not in got.text
    got = wt.pool({"op": "filter", "episode": "e1"}, pdir, box)
    assert a in got.text and b not in got.text
    got = wt.pool({"op": "show", "id": a}, pdir, box)
    assert '"kind": "deferred"' in got.text
    assert got.paths == ({"path": f"/inputs/pool/{a}/entry.json", "mode": "read"},)
    assert wt.pool({"op": "show", "id": "nope"}, pdir, box).form == "unknown_entry"
    assert wt.pool({"op": "drop"}, pdir, box).form == "bad_op"


def test_pool_decide_writes_the_manifest_block_and_open_undoes_it(tmp_path):
    a, b, pdir, box = _staged_pool(tmp_path)
    wt.write_manifest(box, {"items": [], "summary": "s"})
    got = wt.pool(
        {"op": "decide", "id": a, "decision": "retired", "reason": "never needed"},
        pdir,
        box,
    )
    assert got.status == "ok", got.text
    man = json.loads((box / ".pass/manifest.json").read_text())
    assert man["pool"] == {"retired": {a: "never needed"}} and man["summary"] == "s"
    # one decision per entry: a new one replaces the old
    wt.pool({"op": "decide", "id": a, "decision": "superseded", "target": b}, pdir, box)
    man = json.loads((box / ".pass/manifest.json").read_text())
    assert man["pool"] == {"superseded": {a: b}}
    got = wt.pool(
        {
            "op": "decide",
            "id": "new",
            "decision": "defer",
            "reason": "later",
            "episodes": ["e3"],
        },
        pdir,
        box,
    )
    assert json.loads((box / ".pass/manifest.json").read_text())["pool"]["defer"] == [
        {"id": "new", "episodes": ["e3"], "reason": "later"},
    ]
    wt.pool({"op": "decide", "id": a, "decision": "open"}, pdir, box)
    man = json.loads((box / ".pass/manifest.json").read_text())
    assert "superseded" not in man["pool"]
    assert wt.defer_rows(man) == [{"id": "new", "episodes": ["e3"], "reason": "later"}]
    assert wt.without_defer(man) == {"items": [], "summary": "s"}
    # refusals name the domain problem
    bad = wt.pool({"op": "decide", "id": a, "decision": "retired"}, pdir, box)
    assert bad.form == "bad_decision" and "needs a reason" in bad.text
    bad = wt.pool(
        {"op": "decide", "id": "new", "decision": "defer", "reason": "r"},
        pdir,
        box,
    )
    assert bad.form == "bad_decision"
    got = wt.pool(
        {"op": "decide", "id": b, "decision": "merged", "target": "memory.x:f"},
        pdir,
        box,
    )
    assert "lists no item 'memory.x:f' yet" in got.text


# --- structured tool events ----------------------------------------------------------------------------------


def test_every_call_is_one_tool_event_with_status_paths_and_usage(tmp_path):
    """Rev 2: every model call is an execute_code cell (one ``via: model`` row, with the turn's usage on its first
    call), and each tool call a cell makes is a ``via: channel`` row naming that cell, with its status and paths."""
    turns = [
        {
            "role": "assistant",
            "usage": {"prompt_tokens": 100, "completion_tokens": 7},
            "tool_calls": [
                _call("a", "read", {"path": "/inputs/batch_map.json"})["tool_calls"][0],
                _call("b", "grep", {"pattern": "req"})["tool_calls"][0],
                _call("c", "dismiss", {"episode": "e1", "reason": "x"})["tool_calls"][
                    0
                ],
            ],
        },
        _call("d", "read_episode", {"episode": "e1", "parts": ["request"]}),
        _call("e", "read_episode", {"episode": "zz"}),
    ]
    model = Turns(turns)
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=6,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    rows = writer.tool_events
    assert [r["call"] for r in rows] == list(range(len(rows)))
    cells = [r for r in rows if r["via"] == "model"]
    assert all(r["tool"] == "execute_code" and r["cell"] is None for r in cells)
    a, b, c, d, e = cells[:5]
    assert (
        a["usage"]["input"] == 100
        and a["usage"]["output"] == 7
        and a["usage"]["usd"] == "0.001"
    )
    assert a["usage"]["cache_read"] is None  # unreported stays unknown
    assert (
        b["usage"] is None and c["usage"] is None
    )  # usage is on the first call of a turn only

    def made_by(cell):
        return [(r["tool"], r["status"], r["paths"]) for r in rows if r["via"] == "channel" and r["cell"] == cell["call"]]

    assert made_by(a) == [("read", "ok", [{"path": "/inputs/batch_map.json", "mode": "read"}])]
    assert made_by(b) == [("grep", "ok", [{"path": "/inputs", "mode": "search"}])]  # grep's default scope, resolved
    # a tool the writer does not have is no call: the cell's own error says how to list them
    assert made_by(c) == []
    assert "tools has no attribute 'dismiss'" in model.outputs["c"] and "tools.list() lists yours" in model.outputs["c"]
    assert made_by(d) == [("read_episode", "ok", [{"path": "/episodes/e1/request", "mode": "read"}])]
    [refused] = [r for r in rows if r["via"] == "channel" and r["cell"] == e["call"]]
    assert refused["status"] == "refused" and refused["refusal"]["form"] == "unknown_episode"
    assert refused["refusal"]["text_bytes"] == len(model.outputs["e"].encode())
    for r in rows:
        assert (
            r["event"] == "tool_call"
            and r["agent"] == "writer"
            and r["pass_id"] == "p1"
        )
        assert r["utc"].endswith("Z") and r["args_bytes"] > 0
    written = [
        json.loads(line)
        for line in (tmp_path / "out" / "p1" / "tool-events.jsonl")
        .read_text()
        .splitlines()
    ]
    # the tool calls, then the pass's end (its helpers_closed: here none were started); a cell's own calls are
    # written as they are made, before the cell's row, so the file holds the same rows in the order written
    assert sorted(written[:-1], key=lambda r: r["call"]) == rows
    assert written[-1]["event"] == "pass_end" and written[-1]["helpers_closed"] == {}


def test_the_empty_turn_nudge_names_the_real_tools(tmp_path):
    """Rev 2: the writer's one model-API tool is execute_code, and the nudge names it and how the pass ends."""
    model = Turns([{"role": "assistant", "content": "thinking"}])
    _, _, writer = _writer(tmp_path, model, max_calls=3)
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    nudges = [
        m["content"]
        for m in model.seen
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("Use your tools: ")
    ]
    assert nudges and nudges[0] == "Use your tools: execute_code; finish ends the pass."


def test_an_existing_events_file_is_never_written_over(tmp_path):
    folder = tmp_path / "out" / "p1"
    folder.mkdir(parents=True)
    (folder / "tool-events.jsonl").write_text("earlier\n")
    model = Turns([_call("a", "read", {"path": "/inputs"})])
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=3,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    assert (folder / "tool-events.jsonl").read_text() == "earlier\n"
    assert [r["tool"] for r in writer.tool_events[:2]] == ["execute_code", "read"]


def test_the_profile_limits_are_the_enforcing_constants_and_the_texts_state_them():
    from unify.memory import helpers, repair
    from unify.memory.integration import fork, fork_worker

    def limits(agent):
        return {
            lim["limit"]: lim["value"]
            for lim in agent_profiles.profile(agent)["limits"]
        }

    h = limits("helper")
    assert (
        h["cell time"] == helpers.CELL_TIMEOUT_S
        and h["files under /work"] == helpers.WORK_ENTRIES
    )
    assert (
        h["file under /work"] == helpers.WORK_FILE_BYTES
        and h["all files under /work"] == helpers.WORK_TOTAL_BYTES
    )
    assert (
        h["staged files"] == fork_worker.QUOTA_FILES
        and h["staged bytes"] == fork_worker.QUOTA_BYTES
    )
    f = limits("fork")
    assert (
        f["turns"] == fork_worker.TURN_GUARD
        and f["call time"] == fork_worker.HTTP_TIMEOUT_S
    )
    assert f["staged files per episode"] == fork_worker.QUOTA_FILES
    assert f["staged bytes per episode"] == fork_worker.QUOTA_BYTES
    assert limits("gate")["repair rounds (capped pass)"] == repair.REPAIR_ROUNDS
    text = fork.fork_instruction()
    for stated in (
        f"{fork_worker.QUOTA_FILES} files",
        "4 MiB",
        f"{fork_worker.TURN_GUARD} turns",
    ):
        assert stated in text, stated
    assert f"at most {helpers.CELL_TIMEOUT_S:.0f} s" in helpers.helper_brief(False)
