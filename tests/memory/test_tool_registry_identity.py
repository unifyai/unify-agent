"""The shared tool registry renders one definition per tool for every agent.

* Every pair of agents that can call a tool reads the same ``help`` text, the same ``list`` line and the same stub
  function for it (an agent whose arguments differ reads them after one marker line).
* Every agent's model-API tools are ``execute_code`` alone, the same bytes for all.
* Every ``tools.<name>`` a rendered text names is a registered tool.
* No module outside the registry defines a tool (an AST scan).
* The stub a box imports holds no host path, root path, run id, route or environment value, even with distinctive
  values planted in the environment and in ``UNIFY_HOME``.
* With ``UNIFY_TOOLS_IN_CODE`` off, the memory agents' model-API tools, the Tools sections and the refusal codes are
  byte for byte those of the base commit (digests taken there).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import types
from pathlib import Path

import pytest

from unify.common import tool_registry as R
from unify.memory import agent_profiles, helpers, mining, pool, tool_spec

ROOT = Path(__file__).resolve().parents[2]
FULL = frozenset({"pool", "similar", "helper", "edit", "store"})

#: Small stand-ins for the channel client and the shell helper (their real sources are embedded the same way).
CLIENT = '''
class Result(str):
    paths = ()
    next = None
class Refused(Exception):
    pass
class NotRun(Refused):
    pass
class ToolError(Exception):
    pass
class ChannelClosed(Exception):
    pass
CALLS = []
def _call(tool, args):
    CALLS.append((tool, args))
    return Result("%s answered %s" % (tool, sorted(args)))
'''
SHELL = '''
class ShellResult(str):
    returncode = 0
def sh(command, timeout=None):
    return ShellResult("ran " + command)
'''


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.setattr(R, "client_source", lambda: CLIENT)
    monkeypatch.setattr(R, "shell_source", lambda: SHELL)


def _allowed(agent: str) -> tuple[str, ...]:
    return R.allow_list(agent, FULL)


def _load(src: str) -> types.ModuleType:
    mod = types.ModuleType("tools")
    exec(compile(src, "tools.py", "exec"), mod.__dict__)
    return mod


def _functions(src: str) -> dict[str, str]:
    tree = ast.parse(src)
    return {
        n.name: ast.get_source_segment(src, n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }


def _above_marker(text: str) -> str:
    return text.split("\n\n" + R.ROLE_MARKER)[0]


def test_every_pair_of_agents_reads_each_shared_tool_the_same(stand_ins):
    stubs = {a: _functions(R.render_stub(_allowed(a), a)) for a in R.AGENTS}
    pairs = 0
    for a in R.AGENTS:
        for b in R.AGENTS:
            for name in set(_allowed(a)) & set(_allowed(b)):
                spec = R.registry()[name]
                own = {a, b} & set(spec.role_args)
                ha, hb = R.help_text(name, a), R.help_text(name, b)
                assert _above_marker(ha) == _above_marker(hb), (name, a, b)
                if own:
                    continue
                pairs += 1
                assert ha == hb, (name, a, b)
                assert R.list_text([name], a) == R.list_text([name], b), (name, a, b)
                if name != "sh":
                    assert stubs[a][name] == stubs[b][name], (name, a, b)
    assert pairs > 50


def test_the_actor_reads_its_own_arguments_after_the_marker(stand_ins):
    text = R.help_text("finish", "actor")
    assert R.ROLE_MARKER in text and "you call finish(answer)" in text
    assert R.ROLE_MARKER not in R.help_text("finish", "writer")
    mod = _load(R.render_stub(_allowed("actor"), "actor"))
    assert mod.finish("42") == "finish answered ['answer']"
    writer = _load(R.render_stub(_allowed("writer"), "writer"))
    assert writer.finish("done") == "finish answered ['summary']"


def test_every_agent_sends_execute_code_alone_and_the_same_bytes():
    sent = {json.dumps(R.json_tools(a)) for a in R.AGENTS}
    assert len(sent) == 1
    assert [t["function"]["name"] for t in R.json_tools("helper")] == ["execute_code"]
    assert R.EXECUTE_CODE.parameters["additionalProperties"] is False
    for agent in ("writer", "curator", "helper"):
        assert agent_profiles.tool_json(agent, FULL, in_code=True) == R.json_tools(agent)
        assert "execute_code" not in R.allow_list(agent, FULL)
        assert "execute_code" in agent_profiles.profile(agent)["json_tools"]


def test_the_allow_lists_follow_the_profiles_and_the_design():
    actor = set(R.allow_list("actor", FULL))
    teammate = set(R.allow_list("teammate", FULL))
    assert actor == {
        "read", "grep", "install", "send", "messages", "wait", "who", "reply", "stop", "delegate", "finish",
        "store_skills", "sh",
    }  # fmt: skip
    assert teammate == actor - {"delegate", "store_skills"} | {"ask"}
    assert "store_skills" not in R.allow_list("actor", FULL - {"store"})
    assert "delegate" not in R.allow_list("helper", FULL)
    for agent in R.AGENTS:
        for name in agent_profiles.profile(agent)["in_code"]:
            assert name in R.registry(), (agent, name)
            assert agent in R.registry()[name].roles, (agent, name)


def test_every_registered_tool_is_documented_for_code():
    for spec in R.registry().values():
        assert spec.kind in R.KINDS, spec.name
        if spec.name == "execute_code":
            continue
        assert spec.returns and spec.examples and spec.when, spec.name
        if spec.kind != "local":
            module, _, function = spec.handler.partition(":")
            assert module.startswith(("unify.memory.", "unify.actor.")) and function == spec.name, spec.name
    for name in ("read", "grep", "overview", "read_episode", "fixture", "check", "write", "edit", "install"):
        assert R.registry()[name].blocking, name
    for name in ("wait", "ask", "delegate", "stop"):
        assert not R.registry()[name].blocking, name
    assert R.handler_for("actor", "read") == ("unify.actor.tool_handlers:read", True)
    assert R.handler_for("writer", "read") == ("unify.memory.tool_handlers:read", True)


def test_the_moved_constants_still_agree_with_their_owners():
    assert R.POOL_RELATED_PAGE == pool.RELATED_PAGE
    assert R.SIMILAR_MIN_NODES == mining.MIN_NODES


def test_every_tool_named_in_a_rendered_text_is_registered(stand_ins):
    own = {"list", "help", "Refused", "NotRun", "ToolError", "ChannelClosed", "Result"}
    texts = [R.help_text(n, a) for a in R.AGENTS for n in _allowed(a)]
    texts += [R.render_stub(_allowed(a), a) for a in R.AGENTS]
    texts += [
        R.preamble(a, {"environment": "/workspace read-write", "memory": "", "flags": FULL}) for a in R.AGENTS
    ]
    for text in texts:
        assert R.tool_names_in(text) - set(R.registry()) - own == set()


def test_the_preamble_names_only_what_the_agent_has():
    writer = R.preamble("writer", {"environment": "/inputs read-only, /memory read-write", "flags": FULL})
    helper = R.preamble("helper", {"environment": "/work read-write", "flags": FULL})
    actor = R.preamble("actor", {"environment": "/workspace read-write", "flags": FULL})
    assert "tools.finish(summary)" in writer and "tools.done(summary, claims)" in helper
    assert "tools.finish(answer)" in actor and "tools.install" in actor
    assert "tools.install" not in writer and "Messages:" in writer
    assert R.preamble("writer", {"flags": FULL}) == R.preamble("writer", {"flags": FULL})


# --- nothing in a definition outside the registry ---

_SCANNED = ("unify/actor", "unify/common/_async_tool", "unify/memory", "unify/agents")
#: Hand-written tool definitions that remain where the earlier surface still sends them: the loop's response tool
#: (UNIFY_TOOLS_IN_CODE off) and the earlier non-library writer's three templates (the memory pass loop's).
_KNOWN = {"unify/common/_async_tool/loop.py": 1, "unify/memory/writer_pass.py": 3}


def _tool_definitions(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
        fn = keys.get("function")
        if (
            isinstance(keys.get("type"), ast.Constant)
            and keys["type"].value == "function"
            and isinstance(fn, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "parameters" for k in fn.keys)
        ):
            count += 1
    return count


def test_no_module_outside_the_registry_defines_a_tool():
    found = {}
    for folder in _SCANNED:
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # a memory module that imports the registry's class under its old name
            memory_spec = folder == "unify/memory" and any(
                isinstance(n, ast.ImportFrom)
                and n.module in ("tool_spec", "unify.memory.tool_spec")
                and any(a.name == "ToolSpec" for a in n.names)
                for n in ast.walk(tree)
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    callee = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                    if callee == "RegisteredTool" or (memory_spec and callee == "ToolSpec"):
                        found.setdefault(rel, 0)
                        found[rel] += 1
            n = _tool_definitions(tree)
            if n != _KNOWN.get(rel, 0):
                found[rel + " (tool dicts)"] = n
    assert found == {}


# --- C3: the stub holds registry fields only ---

PLANTED_ENV = {
    "PLANTED_ROUTE_URL": "https://planted-route-for-the-stub-test.invalid/v1",
    "PLANTED_TOKEN": "planted-key-for-the-stub-test",
    "PLANTED_RUN": "planted-run-name-for-the-stub-test",
}


def _host_values(tmp_path: Path) -> list[str]:
    home = tmp_path / "planted-run-name-for-the-stub-test" / "attempts" / "a1" / "state"
    return [
        *PLANTED_ENV.values(),
        str(home),
        str(ROOT),
        str(Path.home()),
        tempfile.gettempdir().rstrip("/") + "/",
        os.getcwd(),
    ]


def _assert_clean(text: str, values: list[str]) -> None:
    for value in values:
        assert value not in text, value


def test_the_stub_holds_no_host_path_run_id_route_or_environment_value(stand_ins, monkeypatch, tmp_path):
    for key, value in PLANTED_ENV.items():
        monkeypatch.setenv(key, value)
    home = tmp_path / "planted-run-name-for-the-stub-test" / "attempts" / "a1" / "state"
    home.mkdir(parents=True)
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.setenv("UNIFY_LOCAL_ROOT", str(home / "workspace"))
    values = _host_values(tmp_path)
    for agent in R.AGENTS:
        stub = R.render_stub(_allowed(agent), agent)
        _assert_clean(stub, values)
        _assert_clean(R.preamble(agent, {"environment": "/workspace read-write", "flags": FULL}), values)
        mod = _load(stub)
        assert set(_allowed(agent)) <= set(dir(mod))
        # the actor's stub is exec'd in its box as a module body: the same source works there too
        body: dict = {}
        exec(compile(stub, "<tools>", "exec"), body)
        assert body["read"]("/workspace/a.txt") == "read answered ['path']"
        assert body["sh"]("pwd") == "ran pwd"


def test_the_real_sources_are_embedded_cleanly_once_they_exist(monkeypatch, tmp_path):
    client = ROOT / "unify" / "common" / "tool_channel_client.py"
    shell = ROOT / "unify" / "common" / "box_shell_source.py"
    if not (client.exists() and shell.exists()):
        pytest.skip("the channel client and the shell source are not in this tree yet")
    for key, value in PLANTED_ENV.items():
        monkeypatch.setenv(key, value)
    for agent in R.AGENTS:
        stub = R.render_stub(_allowed(agent), agent)
        compile(stub, "tools.py", "exec")
        _assert_clean(stub, _host_values(tmp_path))


# --- the control arm: byte for byte the base commit's ---

#: sha256 prefixes of ``json.dumps`` of each rendering at the base commit (1f7bd11be), taken on the test worker.
BASE = {
    **dict(
        zip(
            [f"w{m}" for m in range(16)],
            """258ac5560ec2a8a1 04db916337b31b1a 149e521e095cb024 c6fadb799c1b43b3 27f59ef28c979fa7
            2585257601fcc1ef 1327387b3cdf73ac 08093975fbd449db 9c9de39599edcbce 4e08325775530998
            050effef8f203830 4e0d558b1f2ded61 da08d5f79d7f6aff 47f73f528b057b13 1c7ffd82559e91a6
            068b712b2bbbed4c""".split(),
        ),
    ),
    "c0": "fdee13e7c12f2784",
    "c1": "d165178f2b2640c9",
    "h0": "1c16d98e8657b054",
    "h1": "cd893e741a0724f8",
    "sw": "5471803684c288d3",
    "sc": "cbb0b23f813da6d9",
    "r": "5be4b9a8b9c27384",
}
#: Refusal forms added since the base (each under its tool) and the tools that are new.
ADDED_FORMS = {
    "finish": ["answered", "bad_answer"],
    "delegate": ["bad_name", "not_main_agent", "record_full", "short_request", "spawn_off", "too_large"],
    "send": ["record_full", "too_fast", "too_large"],
    "reply": ["record_full", "too_fast", "too_large"],
    "ask": ["record_full", "too_fast", "too_large"],
    "stop": ["not_allowed"],
}
ADDED_TOOLS = ("install", "store_skills")
FLAGS = ("pool", "similar", "helper", "edit")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


def test_with_the_switch_off_the_memory_agents_json_tools_are_the_base_commits(monkeypatch):
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    for m in range(16):
        flags = frozenset(f for i, f in enumerate(FLAGS) if m >> i & 1)
        assert _h(agent_profiles.tool_json("writer", flags)) == BASE[f"w{m}"], flags
    assert _h(agent_profiles.tool_json("curator", frozenset())) == BASE["c0"]
    assert _h(agent_profiles.tool_json("curator", frozenset({"edit"}))) == BASE["c1"]
    assert _h(helpers.helper_tools(False)) == BASE["h0"]
    assert _h(helpers.helper_tools(True)) == BASE["h1"]
    assert _h(tool_spec.tools_section("write", frozenset(FLAGS))) == BASE["sw"]
    assert _h(tool_spec.tools_section("curate", frozenset({"edit"}))) == BASE["sc"]


def test_the_refusal_codes_are_the_base_commits_plus_the_listed_additions():
    codes = tool_spec.refusal_codes()
    for tool in ADDED_TOOLS:
        assert codes.pop(tool)
    for tool, forms in ADDED_FORMS.items():
        assert set(forms) <= set(codes[tool]), tool
        codes[tool] = [f for f in codes[tool] if f not in forms]
    assert _h(codes) == BASE["r"]


def test_the_helper_keeps_its_own_refusal_words_for_read_and_grep():
    assert helpers._own_refusal("read", "outside_box", path="/x") == (
        "refused: /x is outside /inputs, /task, /memory and /work"
    )
    assert tool_spec.refusal("read", "outside_box", path="/x") == "refused: /x is outside /inputs, /memory and /work"
    assert tool_spec.refusal("execute_code", "no_time") == "not run: the pass's wall clock is used up"
    assert tool_spec.refusal("grep", "not_found", path="/x") == "refused: /x does not exist"
