import asyncio
import json
import os
import shutil
from decimal import Decimal

import pytest

import unify.memory_v2.manifest as manifest_module
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action, Cell, CostRow
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sol_pass import SOL_SYSTEM, PassConfig, SolPass, export_for_sol
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_gate import ITEM, KIT, MAN, MOD, TEST

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

_ME = Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read")


def _never(eid):
    raise AssertionError(f"episode {eid} was not requested")


def _e1():
    return _ep(
        episode_id="e1",
        cells=[Cell(0, "print(apis.venmo.me())", "{'user_id': 'u-1'}")],
        actions=[_ME],
    )


def _write(path, text):
    return (
        f"import pathlib; p=pathlib.Path({path!r}); "
        f"p.parent.mkdir(parents=True, exist_ok=True); p.write_text({text!r})"
    )


def _call(cid, name, args):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
        ],
    }


class Script:
    """A scripted model: one execute_code call per cell, then finish; it records every tool result."""

    def __init__(self, cells, usd="0.001"):
        self.cells, self.i, self.usd = cells, 0, usd
        self.outputs: dict[str, str] = {}
        self.first: list[dict] | None = None
        self.tools: list[dict] | None = None

    async def __call__(self, messages, tools):
        if self.first is None:
            self.first, self.tools = [dict(m) for m in messages], tools
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = m["content"]
        if self.i < len(self.cells):
            code = self.cells[self.i]
            self.i += 1
            return _call(f"c{self.i}", "execute_code", {"code": code}), self.usd
        return _call("f", "finish", {"summary": "added venmo.me"}), self.usd


SMOKE = (
    "import memlab.analysis.cells, memlab.analysis.provenance, memlab.analysis.slicing, "
    "memlab.analysis.antiunify, memlab.analysis.shapes, memlab.analysis.transitions, "
    "memlab.analysis.shellout, memlab.analysis.recorded, memlab.replay, memlab.episodes, "
    "memlab.fingerprint; print('OK')"
)
REPLAY = (
    "import json\n"
    "from memlab.episodes import Action\n"
    "from memlab.replay import RecordedEnv\n"
    "ep = json.load(open('/inputs/episodes/e1.json'))\n"
    "env = RecordedEnv([Action(**{k: v for k, v in a.items() if k != 'index'}) for a in ep['actions']])\n"
    "req = json.load(open('/inputs/request.json'))\n"
    "print('REPLAY', env.venmo.me()['user_id'], ep['actions'][0]['index'], req['channel'])\n"
)
READ_ONLY = "try:\n    open('/inputs/x', 'w')\nexcept OSError:\n    print('RO')\n"
# the test command SOL_SYSTEM gives, run as Sol would run it
SUITE = (
    "import subprocess, sys\n"
    "r = subprocess.run([sys.executable, '-m', 'pytest', 'env/venmo/tests', '-q', '-p', 'no:cacheprovider',\n"
    "    '--rootdir', '/memory', '-c', '/dev/null', '--import-mode=importlib'],\n"
    "    env={'PYTHONPATH': '/memory', 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)\n"
    "print(r.stdout[-300:], r.returncode)\n"
)


@needs_bwrap
@pytest.mark.parametrize(
    "per_call, usd, unknown",
    [("0.001", "0.009", 0), ("unknown", "0", 9)],
)
def test_scripted_pass_merges_through_gate(tmp_path, per_call, usd, unknown):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    manifest = {**MAN, "items": [ITEM], "summary": "s"}
    cells = [
        SMOKE,
        REPLAY,
        READ_ONLY,
        _write("/memory/unify_memory_testkit.py", KIT),
        _write("/memory/env/venmo/tests/test_me.py", TEST),
        _write("/memory/env/venmo/__init__.py", MOD),
        SUITE,
        _write("/memory/.pass/manifest.json", json.dumps(manifest)),
    ]
    script = Script(cells, usd=per_call)
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=script,
        config=PassConfig(max_calls=10),
    )
    parent = mem.head()
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    # the toolkit imports inside the box, the episodes replay, and /inputs is read-only
    assert script.outputs["c1"].strip() == "OK", script.outputs["c1"]
    assert script.outputs["c2"].strip() == "REPLAY u-1 0 venmo", script.outputs["c2"]
    assert script.outputs["c3"].strip() == "RO", script.outputs["c3"]
    assert "1 passed" in script.outputs["c7"] and script.outputs["c7"].strip().endswith(
        " 0",
    )
    assert script.first[0] == {"role": "system", "content": SOL_SYSTEM}
    assert {t["function"]["name"] for t in script.tools} == {"execute_code", "finish"}
    assert out.passed, out.reasons
    assert out.calls == 9 and out.usd == usd and out.unknown_cost_calls == unknown
    notes = [f"note: {unknown} unpriced calls"] if unknown else []
    assert out.reasons == notes
    assert out.summary == "added venmo.me"
    assert mem.head() == out.commit != parent
    body = mem.run("log", "-1", "--format=%B", out.commit)
    assert "Pass: p1" in body and "Episode: e1" in body and "Evidence: e1" in body
    tree = mem.run("ls-tree", "-r", "--name-only", out.commit).split()
    assert ".pass/manifest.json" not in tree and not any(
        p.startswith(".pass") for p in tree
    )
    row = ev.db.execute(
        "SELECT passed, usd, reasons FROM passes WHERE pass_id='p1'",
    ).fetchone()
    assert row == (1, usd, json.dumps(notes))


@needs_bwrap
def test_pass_stops_at_call_cap_without_manifest(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"))
    sol = SolPass(
        mem,
        gate,
        ev,
        load=_never,
        model_turn=Script(["print(1)"] * 50),
        config=PassConfig(max_calls=3),
    )
    parent = mem.head()
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p2"))
    assert not out.passed and out.calls == 3 and out.reasons == ["no manifest"]
    assert out.commit is None and out.usd == "0.003" and mem.head() == parent
    row = ev.db.execute(
        "SELECT passed, reasons, usd, candidate FROM passes WHERE pass_id='p2'",
    ).fetchone()
    assert row == (0, json.dumps(["no manifest"]), "0.003", None)


class Costs:
    """A model that answers in text only (no cell runs), at scripted per-call costs."""

    def __init__(self, costs):
        self.costs, self.i = costs, 0

    async def __call__(self, messages, tools):
        usd = self.costs[min(self.i, len(self.costs) - 1)]
        self.i += 1
        return {"role": "assistant", "content": "thinking"}, usd


@pytest.mark.parametrize(
    "costs, calls, usd, unknown",
    [
        # each unknown call reserves max_usd / max_calls = 0.10: 0.40 + 6 x 0.10 reaches the cap
        (["0.40"] + ["unknown"] * 50, 7, "0.40", 6),
        (["0.60"] * 50, 2, "1.20", 0),
        (["unknown"] * 50, 10, "0", 10),
        (["not money"] * 50, 10, "0", 10),
        # never exponent notation (str(Decimal("1E-7")) is "1E-7")
        (["1E-7"] * 50, 10, "0.0000010", 0),
    ],
)
def test_spend_cap_reserves_budget_for_unknown_costs(
    tmp_path,
    costs,
    calls,
    usd,
    unknown,
):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"))
    sol = SolPass(
        mem,
        gate,
        ev,
        load=_never,
        model_turn=Costs(costs),
        config=PassConfig(max_calls=10, max_usd=Decimal("1.00")),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p3"))
    assert out.calls == calls and not out.passed
    # ruling R22: money is a pure decimal string; unpriced calls are counted apart
    assert out.usd == usd and out.unknown_cost_calls == unknown
    notes = [f"note: {unknown} unpriced calls"] if unknown else []
    assert out.reasons == ["no manifest", *notes]
    assert ev.db.execute(
        "SELECT usd, reasons FROM passes WHERE pass_id='p3'",
    ).fetchone() == (usd, json.dumps(["no manifest", *notes]))


def test_recorded_pass_id_is_refused_before_any_model_call(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.record_pass({"pass_id": "p4", "passed": 0, "usd": "0.5"})
    model = Costs(["0.1"])
    sol = SolPass(
        mem,
        Gate(mem, ev, BlobStore(tmp_path / "b")),
        ev,
        load=_never,
        model_turn=model,
        config=PassConfig(),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p4"))
    assert not out.passed and out.calls == 0 and model.i == 0
    assert ev.db.execute("SELECT usd FROM passes WHERE pass_id='p4'").fetchone() == (
        "0.5",
    )


def test_export_for_sol_carries_no_outcome_signal_or_checker_data(tmp_path):
    reason = "CHECKER: failed because the amount paid to Bob was 20, not 25"
    ep = _ep(
        episode_id="e9",
        regime=reason,
        transcript=[{"role": "user", "content": reason}],
        cells=[Cell(0, "print(apis.venmo.me())", "{'user_id': 'u-1'}", error=reason)],
        actions=[
            Action(
                0,
                "venmo",
                "pay",
                [5],
                {"to": "bob"},
                {"ok": True},
                "ok",
                "write",
                None,
            ),
        ],
        memory_diff=reason,
        worktree_diff=reason,
        costs=[CostRow(reason, "m", 1, 1, "0.1")],
        fingerprints={"venmo": {reason: [reason]}},
    )
    # attributes a later schema might add: an outcome, signals, a checker verdict
    ep.outcome = reason
    ep.signals = [{"label": "fail", "reason": reason}]
    ep.actions[0].checker = reason
    out = tmp_path / "episodes"
    out.mkdir()
    export_for_sol(lambda eid: ep, ["e9"], out)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    assert [p.name for p in files] == ["e9.json"]
    assert reason not in files[0].read_text()
    assert json.loads(files[0].read_text()) == {
        "episode_id": "e9",
        "request": ["Pay my Venmo friends back"],
        "cells": [
            {
                "index": 0,
                "code": "print(apis.venmo.me())",
                "output": "{'user_id': 'u-1'}",
            },
        ],
        "actions": [
            {
                "index": 0,
                "cell": 0,
                "channel": "venmo",
                "method": "pay",
                "args": [5],
                "kwargs": {"to": "bob"},
                "response": {"ok": True},
                "status": "ok",
                "effect": "write",
                "error": None,
                "kind": "tool",
            },
        ],
        "memory_channels": ["venmo"],
    }


def test_export_for_sol_names_each_actions_memory_channel_as_the_gate_maps_it(tmp_path):
    ep = _e1()
    ep.actions = [
        Action(
            cell=0,
            channel="env",
            method="say",
            args=["x"],
            kwargs={},
            response="o",
            status="ok",
            kind="dialogue",
        ),
        Action(
            cell=0,
            channel="dialogue:user",
            method="say",
            args=["x"],
            kwargs={},
            response="o",
            status="ok",
            kind="dialogue",
        ),
        Action(
            cell=0,
            channel="shell:git",
            method="run",
            args=["x"],
            kwargs={},
            response=None,
            status="ok",
            kind="dialogue",
        ),
    ]
    export_for_sol(lambda eid: ep, ["e1"], tmp_path)
    row = json.loads((tmp_path / "e1.json").read_text())
    assert row["memory_channels"] == ["env", "dialogue_user", None]
    for a in row["actions"]:  # the brief's rebuild recipe still works on every action
        Action(**{k: v for k, v in a.items() if k != "index"})


def test_export_for_sol_refuses_unsafe_episode_ids(tmp_path):
    with pytest.raises(ValueError):
        export_for_sol(lambda eid: _e1(), ["../e1"], tmp_path)


def test_sol_system_embeds_the_manifest_rules_verbatim():
    doc = manifest_module.__doc__
    rules = doc[doc.index("Manifest rules for consolidators") :].strip()
    assert rules in SOL_SYSTEM
    assert "env/<channel>/tests/test_" in SOL_SYSTEM and '"skeleton"' in SOL_SYSTEM


def test_sol_system_states_that_scope_is_shape_not_observed_values():
    """Spec F3a: refusals name shape only; a value restriction needs a covered environment rejection."""
    assert "Scope is shape, not observed values" in SOL_SYSTEM
    assert "value domains" not in SOL_SYSTEM
    assert "outside the function's scope" not in SOL_SYSTEM
    assert "Bad: `if colour not in" in SOL_SYSTEM
    assert "Good: `if not isinstance(colour" in SOL_SYSTEM
    assert "rejection must then be one of the" in SOL_SYSTEM


def test_sol_system_asks_each_function_to_declare_its_input_from_the_one_constant():
    """Items declare the form of their first argument, in the manifest and as an Input: docstring line."""
    flat = SOL_SYSTEM.replace("\n   ", " ")
    assert manifest_module.describe_input_kinds() in flat
    for name in manifest_module.INPUT_KINDS:
        assert f"{name} (" in flat
    assert "`Input: <form>`" in flat
    assert '"input":"<form>"' in SOL_SYSTEM
    assert "{input_kinds}" not in SOL_SYSTEM


def test_sol_system_lists_the_declared_semantic_types_from_the_one_constant():
    """D21: values are restricted only through the fixed type list, declared in the manifest."""
    flat = SOL_SYSTEM.replace("\n   ", " ")
    assert manifest_module.describe_semantic_types() in flat
    for name in manifest_module.SEMANTIC_TYPES:
        assert f"{name} (" in flat
    assert "month (integer 1–12)" in flat and "percentage (number 0–100)" in flat
    assert "probability (number 0–1)" in flat and "nonneg_money (decimal ≥ 0)" in flat
    assert "currency_code (three uppercase letters, a format)" in flat
    assert "{semantic_types}" not in SOL_SYSTEM


@needs_bwrap
def test_pass_never_follows_links_or_git_files_written_in_the_box(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"))
    planted = tmp_path / "planted.json"  # a host file the box cannot see
    planted.write_text(json.dumps(MAN))
    cells = [
        _write("/memory/.git", "gitdir: /nonexistent\n"),
        _write("/memory/env/venmo/__init__.py", MOD),
        "import os; os.makedirs('/memory/.pass'); "
        f"os.symlink({str(planted)!r}, '/memory/.pass/manifest.json')",
    ]
    sol = SolPass(
        mem,
        gate,
        ev,
        load=_never,
        model_turn=Script(cells),
        config=PassConfig(max_calls=10),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p5"))
    assert not out.passed
    assert any(
        "manifest" in r and "regular file" in r for r in out.reasons
    ), out.reasons
    cand = ev.db.execute("SELECT candidate FROM passes WHERE pass_id='p5'").fetchone()[
        0
    ]
    tree = mem.run("ls-tree", "-r", "--name-only", cand).split()
    assert tree == ["env/venmo/__init__.py"]


def test_model_error_ends_the_pass_and_is_recorded(tmp_path):
    async def boom(messages, tools):
        raise RuntimeError("provider down")

    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    sol = SolPass(
        mem,
        Gate(mem, ev, BlobStore(tmp_path / "b")),
        ev,
        load=_never,
        model_turn=boom,
        config=PassConfig(),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), "p6"))
    assert out.calls == 1 and out.usd == "0" and out.unknown_cost_calls == 1
    assert out.reasons == [
        "model call failed: RuntimeError: provider down",
        "no manifest",
        "note: 1 unpriced calls",
    ]
    assert ev.db.execute("SELECT usd FROM passes WHERE pass_id='p6'").fetchone() == (
        "0",
    )


@pytest.mark.parametrize(
    "costs, usd",
    [
        ([0.0125, 0.0005], "0.0130"),
        ([1e-07], "0.0000001"),
        ([0.01, None], "unknown"),
        ([], "unknown"),
    ],
)
def test_unillm_turn_prices_its_own_events_without_a_model_call(
    monkeypatch,
    costs,
    usd,
):
    import unify.common.llm_client as llm_client
    from unillm.llm_events import LLMEvent, get_llm_event_hook

    from unify.memory_v2.sol_pass import unillm_turn

    built = {}

    class FakeClient:
        messages: list = []

        async def generate(self, *, messages, tools, tool_choice, stateful):
            assert (
                stateful and tool_choice == "auto" and messages[0]["role"] == "system"
            )
            # delivered to the context-local hook only (as unillm does), never to process-wide listeners
            hook = get_llm_event_hook()
            for c in costs:  # e.g. a postprocessing retry emits a second event
                hook(LLMEvent(request={}, provider_cost=c, origin="memory_v2.sol"))
            hook(LLMEvent(request={}, provider_cost=9.0, origin="elsewhere"))
            self.messages = [
                *messages,
                {"role": "assistant", "content": "hi", "tool_calls": None},
            ]

    def fake_new(model, **kw):
        built.update(model=model, **kw)
        return FakeClient()

    monkeypatch.setattr(llm_client, "new_llm_client", fake_new)
    turn = unillm_turn("openai/gpt-6-sol", "low")
    msg, got = asyncio.run(
        turn(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            [],
        ),
    )
    assert built == {
        "model": "openai/gpt-6-sol@openrouter",
        "origin": "memory_v2.sol",
        "reasoning_effort": "low",
        "stateful": True,
    }
    assert msg["content"] == "hi" and got == usd


class Turns:
    """A model that replays scripted assistant messages, then finishes."""

    def __init__(self, messages, usd="0.001"):
        self.queue, self.usd = list(messages), usd
        self.outputs: dict[str, str] = {}
        self.seen: list[dict] = []

    async def __call__(self, messages, tools):
        self.seen = messages
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = m["content"]
        if self.queue:
            return self.queue.pop(0), self.usd
        return _call("fin", "finish", {"summary": "done"}), self.usd


def _sol(tmp_path, model, **cfg):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"))
    cfg.setdefault("max_calls", 10)
    sol = SolPass(
        mem,
        gate,
        ev,
        load=_never,
        model_turn=model,
        config=PassConfig(**cfg),
    )
    return mem, ev, sol


def _run(sol, pass_id="px"):
    return asyncio.run(sol.run(PassRequest("incremental", "venmo", [], False), pass_id))


DEEP = "[" * 200000 + "]" * 200000


@needs_bwrap
def test_hostile_cells_and_arguments_never_raise_on_the_host(tmp_path):
    big = (
        "x = 1\n" * 40000 + "print('BIG', x)\n"
    )  # 240 KB: over the 128 KiB argv string limit
    calls = [
        _call("big", "execute_code", {"code": big}),
        _call("nul", "execute_code", {"code": "print(1)\0"}),
        _call(
            "utf8",
            "execute_code",
            {
                "code": "import sys; sys.stdout.buffer.write(b'A\\xff\\xfeB'); sys.stdout.flush()",
            },
        ),
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "deep",
                    "type": "function",
                    "function": {"name": "execute_code", "arguments": DEEP},
                },
            ],
        },
        _call(
            "deepman",
            "execute_code",
            {"code": _write("/memory/.pass/manifest.json", DEEP)},
        ),
    ]
    model = Turns(calls)
    mem, ev, sol = _sol(tmp_path, model)
    out = _run(sol)
    assert model.outputs["big"].strip() == "BIG 1"
    assert "null bytes" in model.outputs["nul"]
    assert model.outputs["utf8"].strip() == "A\ufffd\ufffdB"
    assert model.outputs["deep"].startswith("unreadable arguments: RecursionError")
    assert not out.passed
    assert any(
        r.startswith("manifest: not valid JSON (RecursionError") for r in out.reasons
    ), out.reasons
    assert ev.pass_exists("px")


@needs_bwrap
def test_chmod_zero_entries_are_unlocked_measured_and_mirrored(tmp_path):
    code = (
        "import os\n"
        "os.makedirs('/memory/env/venmo/hidden')\n"
        "open('/memory/env/venmo/hidden/x.py', 'w').write('x = 1')\n"
        "os.chmod('/memory/env/venmo/hidden', 0)\n"
        "open('/memory/env/venmo/secret.md', 'w').write('s')\n"
        "os.chmod('/memory/env/venmo/secret.md', 0)\n"
        "open('/memory/env/venmo/NOTES.md', 'w').write('# Venmo\\n')\n"
        "os.makedirs('/memory/.pass')\n"
        "open('/memory/.pass/manifest.json', 'w').write('{}')\n"
    )
    mem, ev, sol = _sol(tmp_path, Script([code]))
    out = _run(sol)
    assert not out.passed  # the gate still decides; nothing crashed on the host
    assert not any(
        "left out" in r or "unreadable" in r for r in out.reasons
    ), out.reasons
    cand = ev.db.execute("SELECT candidate FROM passes WHERE pass_id='px'").fetchone()[
        0
    ]
    # the host restored its own access, so the gate sees (and refuses) exactly what Sol wrote
    assert mem.run("ls-tree", "-r", "--name-only", cand).split() == [
        "env/venmo/NOTES.md",
        "env/venmo/hidden/x.py",
        "env/venmo/secret.md",
    ]


@needs_bwrap
@pytest.mark.parametrize(
    "code, why",
    [
        ("open('/memory/big.txt', 'wb').write(b'x' * (1024**2 + 1))", "per file"),
        (
            "import os\nfor i in range(2001):\n    open(f'/memory/f{i}', 'w').close()",
            "files and directories",
        ),
        (
            "import os\nos.makedirs('/memory/env/a')\n"
            "for i in range(21):\n    open(f'/memory/env/a/f{i}', 'wb').write(b'x' * 1024**2)",
            "bytes under /memory",
        ),
    ],
)
def test_write_quota_refuses_the_pass_without_a_commit(tmp_path, code, why):
    later = _write("/memory/.pass/manifest.json", json.dumps(MAN))
    model = Script([code, later])
    mem, ev, sol = _sol(tmp_path, model)
    parent = mem.head()
    out = _run(sol)
    assert not out.passed and out.commit is None and out.calls == 1
    assert any(
        r.startswith("over quota") and why in r for r in out.reasons
    ), out.reasons
    row = ev.db.execute(
        "SELECT passed, candidate FROM passes WHERE pass_id='px'",
    ).fetchone()
    assert row == (0, None) and mem.head() == parent


@needs_bwrap
def test_cell_file_size_is_limited_below_run_confineds(tmp_path):
    from unify.memory_v2.sol_pass import CELL_FSIZE_BYTES

    code = "import resource\n" "print(resource.getrlimit(resource.RLIMIT_FSIZE)[0])\n"
    model = Script([code])
    mem, ev, sol = _sol(tmp_path, model)
    _run(sol)
    assert model.outputs["c1"].strip() == str(CELL_FSIZE_BYTES)


@needs_bwrap
def test_calls_after_finish_are_skipped_and_missing_ids_are_synthesised(tmp_path):
    turn = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": "print('first')"}),
                },
            },
            {
                "id": "f",
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": json.dumps({"summary": "s"}),
                },
            },
            {
                "id": "late",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": "open('/memory/late', 'w')"}),
                },
            },
        ],
    }
    model = Turns([turn])
    mem, ev, sol = _sol(tmp_path, model)
    out = _run(sol)
    # the list the model was last given is the pass's transcript, final tool replies included
    tool_msgs = [m for m in model.seen if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1_0", "f", "late"]
    assert turn["tool_calls"][0]["id"] == "call_1_0"
    assert tool_msgs[0]["content"].strip() == "first"
    assert tool_msgs[2]["content"] == "not run: pass finished"
    assert out.calls == 1 and out.summary == "s" and out.reasons == ["no manifest"]


def test_pass_deadline_stops_a_slow_model(tmp_path):
    class Slow:
        calls = 0

        async def __call__(self, messages, tools):
            Slow.calls += 1
            await asyncio.sleep(0.4)
            return {"role": "assistant", "content": "thinking"}, "0.01"

    mem, ev, sol = _sol(tmp_path, Slow(), max_calls=40, deadline_s=1.0)
    out = _run(sol)
    assert 2 <= out.calls <= 4
    assert any("pass deadline of 1.0 s reached" in r for r in out.reasons), out.reasons
    assert ev.pass_exists("px")


def test_cancelled_pass_is_recorded(tmp_path):
    started = asyncio.Event()

    async def hang(messages, tools):
        started.set()
        await asyncio.sleep(3600)

    mem, ev, sol = _sol(tmp_path, hang)

    async def main():
        task = asyncio.create_task(
            sol.run(PassRequest("incremental", "venmo", [], False), "pc"),
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    usd, reasons = ev.db.execute(
        "SELECT usd, reasons FROM passes WHERE pass_id='pc'",
    ).fetchone()
    assert usd == "0" and json.loads(reasons)[0].startswith(
        "pass error: CancelledError",
    )


@needs_bwrap
def test_an_unreadable_memory_root_is_unlocked_and_measured(tmp_path):
    code = (
        "import os\n"
        "os.makedirs('/memory/env/venmo')\n"
        "open('/memory/env/venmo/NOTES.md', 'w').write('# Venmo\\n')\n"
        "os.makedirs('/memory/.pass')\n"
        "open('/memory/.pass/manifest.json', 'w').write('{}')\n"
        "os.chmod('/memory/env/venmo', 0)\n"
        "os.chmod('/memory', 0)\n"
    )
    mem, ev, sol = _sol(tmp_path, Script([code]))
    out = _run(sol)
    assert not out.passed and ev.pass_exists("px")
    cand = ev.db.execute("SELECT candidate FROM passes WHERE pass_id='px'").fetchone()[
        0
    ]
    # nothing was left out: the host made every directory owner-readable before measuring and mirroring
    assert mem.run("ls-tree", "-r", "--name-only", cand).split() == [
        "env/venmo/NOTES.md",
    ]
    assert not any("left out" in r for r in out.reasons), out.reasons


@needs_bwrap
def test_a_parsable_but_deep_manifest_is_refused_before_the_gate(tmp_path):
    deep = json.dumps({"items": [[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]})
    mem, ev, sol = _sol(tmp_path, Script([_write("/memory/.pass/manifest.json", deep)]))
    out = _run(sol)
    assert not out.passed
    assert "manifest: nested deeper than 16 levels" in out.reasons, out.reasons


@needs_bwrap
@pytest.mark.parametrize("mode", [0, 0o444, 0o111])
def test_quota_cannot_be_hidden_under_an_unreadable_directory(tmp_path, mode):
    code = (
        "import os\n"
        "os.makedirs('/memory/hide')\n"
        "for i in range(25):\n"
        "    open(f'/memory/hide/f{i}', 'wb').write(b'x' * 1024**2)\n"
        "os.makedirs('/memory/.pass')\n"
        f"open('/memory/.pass/manifest.json', 'w').write({json.dumps(json.dumps(MAN))})\n"
        f"os.chmod('/memory/hide', {mode})\n"
    )
    model = Script([code])
    mem, ev, sol = _sol(tmp_path, model)
    parent = mem.head()
    out = _run(sol)
    assert not out.passed and out.commit is None and out.calls == 1
    assert any(r.startswith("over quota: more than") for r in out.reasons), out.reasons
    row = ev.db.execute(
        "SELECT passed, candidate FROM passes WHERE pass_id='px'",
    ).fetchone()
    assert row == (0, None) and mem.head() == parent


def test_measure_refuses_entries_that_stay_unreadable(tmp_path, monkeypatch):
    import unify.memory_v2.sol_pass as sp

    (tmp_path / "d").mkdir()
    real = os.scandir

    def deny(path):
        if str(path).endswith("/d"):
            raise PermissionError(13, "denied")
        return real(path)

    monkeypatch.setattr(sp.os, "scandir", deny)
    assert sp._measure(tmp_path) == "over quota: unreadable entries ['d']"


@needs_bwrap
def test_pass_notes_reach_the_recorded_row_on_the_gate_path(tmp_path):
    class Model:
        i = 0

        async def __call__(self, messages, tools):
            Model.i += 1
            if Model.i == 1:
                code = _write("/memory/.pass/manifest.json", "{}")
                return _call("w", "execute_code", {"code": code}), "unknown"
            raise RuntimeError("provider down")

    mem, ev, sol = _sol(tmp_path, Model())
    out = _run(sol)
    assert ev.db.execute("SELECT candidate FROM passes WHERE pass_id='px'").fetchone()[
        0
    ]
    row = json.loads(
        ev.db.execute("SELECT reasons FROM passes WHERE pass_id='px'").fetchone()[0],
    )
    assert row[-2:] == [
        "model call failed: RuntimeError: provider down",
        "note: 2 unpriced calls",
    ]
    assert out.reasons == row
    assert out.unknown_cost_calls == 2 and out.usd == "0"
