"""What the writer is told: briefs per pass type and configuration, the first message, and what the transcript
keeps. Lists that grow (library index, pool, use table, drafts, curate reasons) are files the brief points to; a
sentence that relies on an optional tool appears only when that tool is registered for the role; no economy cue or
reading-quota word reaches the writer; the transcript keeps the brief and the tool JSON by hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from unify.memory import library_prompts as lp
from unify.memory import tool_spec, writer_pass
from unify.memory.tool_spec import ToolSpec
from unify.memory.trigger import PassRequest
from unify.memory.usage import LIBRARY_COPY, writer_usage_record

_SCHEMA = {"type": "object", "properties": {}}
_WRITE_ONLY_TOOLS = ("pool", "similar", "delegate", "replay", "land_capture")


def _spec(name, roles, requires=None):
    return ToolSpec(
        name=name,
        description=f"The {name} tool. More text.",
        parameters=_SCHEMA,
        # the registry names the agents (writer, curator, helper), not the earlier pass roles
        roles=tuple(sorted({"write": "writer", "curate": "curator"}.get(r, r) for r in roles)),
        requires=requires,
    )


#: A registry with every tool the contracts name, so the briefs render as they will once all tools exist.
_FULL = (
    _spec("read", {"write", "curate", "helper"}),
    _spec("grep", {"write", "curate", "helper"}),
    _spec("overview", {"write", "curate", "helper"}),
    _spec("read_episode", {"write", "curate", "helper"}),
    _spec("similar", {"write", "helper"}),
    _spec("pool", {"write"}, requires="pool"),
    _spec("fixture", {"write", "curate"}),
    _spec("replay", {"write"}),
    _spec("land_capture", {"write"}),
    _spec("delegate", {"write"}),
    _spec("execute_code", {"write", "curate", "helper"}),
    _spec("check", {"write", "curate"}),
    _spec("finish", {"write", "curate"}),
)


@pytest.fixture(params=["empty", "full"])
def registry(request, monkeypatch):
    specs = () if request.param == "empty" else _FULL
    monkeypatch.setattr(tool_spec, "registry", lambda: specs)
    return specs


def _briefs():
    out = {}
    for fork in (False, True):
        for pool in (False, True):
            out[f"write fork={fork} pool={pool}"] = lp.write_brief_now(
                staging=fork,
                pool=pool,
            )
    out["curate"] = lp.curate_brief_now()
    return out


def _lists(text):
    """Runs of consecutive list lines ("- ..." or table rows "a | b") outside the Tools section (rendered once
    per run from the registry), each as (items, bytes)."""
    if "# Tools\n" in text:
        head, rest = text.split("# Tools\n", 1)
        text = head + ("\n# " + rest.split("\n# ", 1)[1] if "\n# " in rest else "")
    runs, cur = [], []
    for line in text.splitlines():
        if line.startswith("- ") or line.count(" | ") >= 2:
            cur.append(line)
        else:
            if cur:
                runs.append((len(cur), len("\n".join(cur).encode())))
            cur = []
    if cur:
        runs.append((len(cur), len("\n".join(cur).encode())))
    return runs


def _first(kind="batched", eids=("e1", "e2")):
    stub = SimpleNamespace(cfg=SimpleNamespace(deadline_s=2700.0))
    req = PassRequest(kind, None, list(eids), True, experience_tokens=116767)
    return writer_pass.WriterPass._first_message_library_mode(stub, req, "p1")


def test_no_inline_list_is_longer_than_the_bound(registry):
    for name, text in {
        **_briefs(),
        "first": _first(),
        "first curate": _first("curate"),
    }.items():
        for items, size in _lists(text):
            assert items <= lp.MAX_INLINE_LIST, name
            # a list of stable text may be long; only the first message's must also be short in bytes
            if name.startswith("first"):
                assert size <= lp.MAX_INLINE_LIST_BYTES, name


def test_no_banned_word_reaches_the_writer(registry):
    for name, text in {
        **_briefs(),
        "first": _first(),
        "first curate": _first("curate"),
    }.items():
        low = text.lower()
        for word in lp.BANNED_WORDS:
            assert word not in low, (name, word)
        for gone in (
            "dismiss",
            "only this tool counts as reading",
            "signals`",
            "items_used",
            "/inputs/s0",
        ):
            assert gone not in low, (name, gone)


def test_the_first_message_points_and_is_stable_apart_from_ids_and_the_wall_clock():
    one = _first(eids=("e1", "e2"))
    two = _first(eids=("e3", "e4"))
    assert one.replace("e1", "X").replace("e2", "Y") == two.replace("e3", "X").replace(
        "e4",
        "Y",
    )
    assert "2700 s" in one and "/inputs/batch_map.json" in one
    for inlined in (
        "Memory index",
        "Candidate pool",
        "Open drafts",
        "Library use since",
        "{",
    ):
        assert inlined not in one
    many = _first(eids=tuple(f"e{i:02d}" for i in range(lp.MAX_INLINE_LIST + 1)))
    assert (
        "e00" not in many
        and f"over {lp.MAX_INLINE_LIST + 1} recorded episodes." in many
    )
    assert len(one.encode()) <= lp.MAX_INLINE_LIST_BYTES
    cur = _first("curate")
    assert "/inputs/curate/trigger.json" in cur and "runs because" not in cur


def test_optional_inputs_appear_only_when_present(registry):
    b = _briefs()
    for name, text in b.items():
        fork = "fork=True" in name
        pool = "pool=True" in name
        assert ("/inputs/staging" in text) == fork, name
        assert ("`staging`" in text) == fork, name
        assert ("/inputs/pool" in text) == pool, name
        assert ("pool_entries" in text) == pool, name
    curate = b["curate"]
    for denied in ("/inputs/staging", "/inputs/pool", "batch_map", "/inputs/helpers"):
        assert denied not in curate
    for tool in _WRITE_ONLY_TOOLS:
        assert f"`{tool}`" not in curate
    assert '"kept": {' in curate and "never a failure" in curate


def test_every_named_path_is_a_staged_input_or_the_library(registry):
    allowed = re.compile(
        r"^/(inputs/(staging|captures|helpers|pool|drafts|batch_map\.json|episodes|shared|usage|gate|curate)|memory|outputs)\b",
    )
    for name, text in _briefs().items():
        for path in re.findall(r"(?<![\w.])/(?:inputs|memory|outputs)[\w./<>-]*", text):
            assert allowed.match(path), (name, path)


def test_the_brief_states_no_git_history_and_the_curate_rollback_files(registry):
    b = _briefs()
    for name, text in b.items():
        assert "no git history" in text, name
        assert "git log" not in text and "git show" not in text, name
    assert "/inputs/curate/rollback/" in b["curate"]
    assert (
        "same_shape.json" in b["curate"]
        and "/inputs/gate/previous.md" not in b["curate"]
    )


def test_limits_are_stated_once_as_information(registry):
    for name, text in _briefs().items():
        assert text.count("# Limits\n") == 1, name
        assert f"after {writer_pass.STEP_GUARD} model and check calls" in text
        assert "none is a target" in text
        assert "of 2" not in text  # no invented repair-round count
        assert "There is no set number of repair rounds" in text


def test_the_gate_section_defines_every_check_it_names(registry):
    from unify.memory.gate import CHECKS

    for name, text in _briefs().items():
        gate = text.split("# The gate\n", 1)[1].split("\n# ", 1)[0]
        for check in CHECKS:
            assert f"{check}, " in gate, (name, check)


def test_trust_and_field_contradictions_are_gone(registry):
    write = lp.write_brief_now(staging=True, pool=True)
    # the manifest parser reads field_types per fixture field in library mode, so the brief documents that form
    assert '`{"<fixture path>#<field>": "<type>"}`' in write
    assert (
        "in this order of trust" in write
        and "A value you make up is not trusted" in write
    )
    assert "`structural_events`" in write and "`end_parts`" in write
    # the done condition is a decision, not a count
    assert (
        "when you have built the best library items you can from this batch, with the proposals as material and the episodes as evidence, and every item you wrote "
        "has passed the gate or is kept as a draft"
    ) in write


def test_the_brief_describes_every_batch_map_key():
    write = lp.write_brief(
        limits=lp.limits_now(),
        mutation_min=lp.Decimal("0.6"),
        sample_k=8,
        lint_min=4,
        pytest_env={},
    )
    for key, _ in lp.row_keys(staging=True, pool=True):
        assert f"`{key}` (" in write, key


def test_pool_kinds_follow_the_pool_module():
    from unify.memory.pool import KINDS

    text = lp._pool_kinds()
    for kind in KINDS:
        assert f"`{kind}`" in text


# --- what the transcript keeps -----------------------------------------------------------------------------


def _messages(brief="BRIEF " * 50, first="FIRST " * 4000):
    return [
        {"role": "system", "content": brief},
        {"role": "user", "content": first},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "[compacted: stub]"},
    ]


def test_capture_stores_brief_and_tools_by_hash_and_keeps_what_was_seen():
    stored: dict[str, bytes] = {}

    def put(data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        stored[sha] = data
        return sha

    msgs = _messages()
    tools = [{"type": "function", "function": {"name": "read"}}]
    lines = [
        json.loads(x)
        for x in writer_pass.transcript_lines(
            msgs,
            "p1",
            tools=tools,
            originals={3: "the whole result before compaction"},
            store=put,
        )
    ]
    head = lines[0]["header"]
    brief_sha = hashlib.sha256(msgs[0]["content"].encode()).hexdigest()
    assert (
        head["brief_sha256"] == brief_sha
        and stored[brief_sha] == msgs[0]["content"].encode()
    )
    assert (
        head["tools_sha256"] in stored
        and json.loads(stored[head["tools_sha256"]]) == tools
    )
    assert "brief" not in head  # by hash only when a store holds it
    rows = {r["i"]: r for r in lines[1:]}
    assert rows[0]["message"]["content"] == f"(brief sha256 {brief_sha})"
    # the first user message is kept whole, beyond the per-string cut
    assert rows[1]["message"]["content"] == msgs[1]["content"]
    assert len(msgs[1]["content"]) > writer_pass.TRANSCRIPT_STRING_CHARS
    assert rows[3]["message"]["content"] == "the whole result before compaction"
    assert rows[3]["compacted"] is True


def test_capture_without_a_store_carries_each_text_once():
    brief = "a brief seen once " + hashlib.sha256(b"unique").hexdigest()
    one = json.loads(writer_pass.transcript_lines(_messages(brief=brief), "p1")[0])
    two = json.loads(writer_pass.transcript_lines(_messages(brief=brief), "p2")[0])
    assert (
        one["header"]["brief"] == brief
        and one["header"]["brief_stored"] == "transcript"
    )
    assert (
        "brief" not in two["header"]
        and two["header"]["brief_stored"] == "earlier transcript"
    )
    assert one["header"]["brief_sha256"] == two["header"]["brief_sha256"]


# --- use records the writer reads ----------------------------------------------------------------------------


def test_a_use_record_loses_every_host_path():
    root = "/home/someone/.local/share/run-x/memory-checkout"
    rec = {
        "export_roots": [root],
        "imported": ["memory.text.dates"],
        "cell_status": [{"where": f"{root}/memory/text/dates.py", "status": "ok"}],
    }
    out = writer_usage_record(rec)
    text = json.dumps(out)
    assert (
        "export_roots" not in out and root not in text and "/home/someone" not in text
    )
    assert out["imported"] == ["memory.text.dates"]
    assert out["cell_status"][0]["where"] == f"{LIBRARY_COPY}/memory/text/dates.py"


def test_the_briefs_pytest_command_has_the_gates_root_and_working_directory():
    import inspect

    from unify.memory import sandbox_run

    gate_src = inspect.getsource(sandbox_run)
    assert (
        '"--rootdir",\n                cwd,' in gate_src
    )  # the gate roots pytest at its working directory, /memory
    for name, text in _briefs().items():
        assert '"--rootdir", "/memory"], cwd="/memory"' in text, name
