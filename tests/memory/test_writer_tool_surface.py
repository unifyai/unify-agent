"""The writer's tool surface: tool JSON rendered from the one spec through the agent profiles, the curator's own
tool set, the pool tool, structured tool events and the removed reading rule."""

import asyncio
import json

import pytest

from unify.memory import agent_profiles, pool, tool_spec
from unify.memory import writer_tools as wt
from unify.memory.integration import switch
from unify.memory.writer_pass import PassConfig, writer_tools
from unify.memory.trigger import PassRequest
from unify.settings import SETTINGS
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import EPS

CURATE_TOOLS = {
    "read",
    "grep",
    "overview",
    "read_episode",
    "fixture",
    "execute_code",
    "check",
    "finish",
    # every agent on the pass's record
    "send",
    "messages",
    "wait",
    "who",
    "reply",
    "stop",
}


def test_the_tool_json_is_the_spec_render():
    for pool_on in (False, True):
        flags = frozenset({"pool"} if pool_on else ())
        got = writer_tools(library_mode=True, pool=pool_on)
        want = [s.as_tool() for s in tool_spec.tools_for("write", flags)]
        assert got == want
        assert {t["function"]["name"] for t in got} == set(
            agent_profiles.tool_names("writer", flags),
        )
    names = [s.name for s in tool_spec.registry()]
    assert len(names) == len(set(names)) and "dismiss" not in names


def test_the_curator_gets_only_its_own_tools():
    for pool_on in (False, True):
        got = {
            t["function"]["name"]
            for t in writer_tools(library_mode=True, pool=pool_on, role="curate")
        }
        assert got == CURATE_TOOLS
    got = {
        t["function"]["name"]
        for t in writer_tools(library_mode=True, role="curate", edit_tools=True)
    }
    assert got == CURATE_TOOLS | {"write", "edit"}
    row = agent_profiles.profile("curator")
    # write and edit are named in the row and exist only with the edit switch on
    assert set(row["tools"]) == CURATE_TOOLS | {"write", "edit"}
    for denied in ("pool", "similar", "replay", "delegate"):
        assert denied in row["denied"] and denied not in row["tools"]


def test_every_profile_tool_exists_in_the_spec_and_rows_have_the_interface_shape():
    known = {s.name for s in tool_spec.registry()}
    agents = [r["agent"] for r in agent_profiles.AGENT_PROFILES]
    assert agents == ["actor", "fork", "writer", "curator", "helper", "gate", "harness"]
    modes = {r["agent"]: r["mode"] for r in agent_profiles.AGENT_PROFILES}
    assert modes == {
        "actor": "consumer",
        "fork": "fork",
        "writer": "write",
        "curator": "curate",
        "helper": "helper",
        "gate": "repair",
        "harness": "inputs",
    }
    keys = {
        "agent",
        "mode",
        "instructions",
        "tools",
        "mounts",
        "denied",
        "consumes",
        "produces",
        "limits",
        "inlined",
        "capabilities",
        "absent_paths",
        "expectations",
        "fields",
        "emits",
        "pointers",
    }
    for row in agent_profiles.AGENT_PROFILES:
        assert keys <= set(row), row["agent"]
        assert all(m["mode"] in ("ro", "rw") for m in row["mounts"])
        if (
            row["agent"] == "fork"
        ):  # the actor's schemas, kept for the prompt cache and never executed
            assert row["tools_executed"] is False
            continue
        for name in row["tools"]:
            assert name in known or name in (
                "similar",
                "delegate",
                "stage",
                "done",
            ), name
        assert not set(row["tools"]) & set(row["denied"])
        for exp in row["expectations"]:
            assert exp["tool"] in row["tools"] or exp["tool"].startswith("memory."), exp
        for ptr in row["pointers"]:
            assert ptr["tool"] in row["tools"] or ptr["tool"].startswith("memory."), ptr
        for lim in row["limits"]:
            assert lim["value"] not in (None, "")
    # the spec's roles and the profiles agree: a spec tool for a role is named in that role's profile
    for agent, role in agent_profiles.SPEC_ROLE.items():
        spec_names = {s.name for s in tool_spec.tools_for(role, frozenset({"pool"}))}
        assert spec_names == set(
            agent_profiles.tool_names(agent, frozenset({"pool"})),
        ), agent


def test_every_hand_off_has_a_producer_and_a_consumer():
    rows = agent_profiles.AGENT_PROFILES
    agents = {r["agent"] for r in rows}
    produced = {(p["artifact"], p["consumer"]) for r in rows for p in r["produces"]}
    consumed = {(c["artifact"], r["agent"]) for r in rows for c in r["consumes"]}
    assert consumed == produced
    assert {c for _, c in produced} <= agents


def test_the_call_record_is_a_gate_internal_hand_off():
    """The gate's test runs write the call record in its box; the gate consumes it, the tests the writer and the
    curator wrote produce it, and no model-facing agent is handed it."""
    from unify.memory import example_calls

    rows = {r["agent"]: r for r in agent_profiles.AGENT_PROFILES}
    record = {"artifact": "call record", "path": example_calls.CALLS_OUT}
    assert record in rows["gate"]["consumes"]
    for agent in ("writer", "curator"):
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
    # the writer's and curator's own boxes are unchanged: they never see the record
    for agent in ("writer", "curator"):
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


def test_the_reading_setting_accepts_empty_and_optional_and_refuses_required():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_COVERAGE == ""
    assert switch.parse_library_coverage("") == "optional"
    assert switch.parse_library_coverage(" Optional ") == "optional"
    assert switch.PARSERS[switch.LIBRARY_COVERAGE] is switch.parse_library_coverage
    with pytest.raises(ValueError, match="required is no longer supported"):
        switch.parse_library_coverage("required")
    with pytest.raises(ValueError, match="UNIFY_MEMORY_LIBRARY_COVERAGE"):
        switch.parse_library_coverage("off")
    assert PassConfig().coverage_rule == "optional"


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
        library_mode=True,
        max_calls=6,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    rows = writer.tool_events
    by_call = {i: r for i, r in enumerate(rows)}
    assert [r["call"] for r in rows] == list(range(len(rows)))
    a, b, c, d, e = (by_call[i] for i in range(5))
    assert (a["tool"], a["status"], a["paths"]) == (
        "read",
        "ok",
        [{"path": "/inputs/batch_map.json", "mode": "read"}],
    )
    assert (
        a["usage"]["input"] == 100
        and a["usage"]["output"] == 7
        and a["usage"]["usd"] == "0.001"
    )
    assert a["usage"]["cache_read"] is None  # unreported stays unknown
    assert (
        b["usage"] is None and c["usage"] is None
    )  # usage is on the first call of a turn only
    assert b["paths"] == [
        {"path": "/inputs", "mode": "search"},
    ]  # grep's default scope, resolved
    assert c["status"] == "refused" and c["refusal"]["form"] == "unknown_tool"
    assert c["refusal"]["text_bytes"] == len(model.outputs["c"].encode())
    assert "your tools: read, grep, overview, read_episode" in model.outputs["c"]
    assert d["paths"] == [{"path": "/episodes/e1/request", "mode": "read"}]
    assert e["status"] == "refused" and e["refusal"]["form"] == "unknown_episode"
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
    # the tool calls, then the pass's end (its helpers_closed: here none were started)
    assert written[:-1] == rows
    assert written[-1]["event"] == "pass_end" and written[-1]["helpers_closed"] == {}


def test_the_empty_turn_nudge_names_the_real_tools(tmp_path):
    model = Turns([{"role": "assistant", "content": "thinking"}])
    _, _, writer = _writer(tmp_path, model, library_mode=True, max_calls=3)
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    nudges = [
        m["content"]
        for m in model.seen
        if m.get("role") == "user"
        and str(m.get("content", "")).startswith("Use your tools: ")
    ]
    assert nudges and "read_episode" in nudges[0] and "fixture" in nudges[0]
    assert "dismiss" not in nudges[0]


def test_an_existing_events_file_is_never_written_over(tmp_path):
    folder = tmp_path / "out" / "p1"
    folder.mkdir(parents=True)
    (folder / "tool-events.jsonl").write_text("earlier\n")
    model = Turns([_call("a", "read", {"path": "/inputs"})])
    _, _, writer = _writer(
        tmp_path,
        model,
        library_mode=True,
        max_calls=3,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    assert (folder / "tool-events.jsonl").read_text() == "earlier\n"
    assert writer.tool_events and writer.tool_events[0]["tool"] == "read"


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
        "900 s",
        f"{fork_worker.TURN_GUARD} turns",
    ):
        assert stated in text, stated
    assert f"at most {helpers.CELL_TIMEOUT_S:.0f} s" in helpers.helper_brief(False)
