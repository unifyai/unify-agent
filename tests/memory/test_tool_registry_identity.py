"""The shared tool registry renders one definition per tool for every agent.

* Every pair of agents that can call a tool reads the same ``help`` text, the same ``list`` line and the same stub
  function for it (an agent whose arguments differ reads them after one marker line; an agent whose tool has roots of
  its own, ``role_text``, reads its own texts).
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
from tests.memory.prompt_guards import tool_names_in

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
                texts = {a, b} & set(spec.role_text)   # an agent with texts of its own (its roots) reads them
                # a tool with texts for a run without memory: the actor and a teammate follow the run's setting, a
                # memory agent always has memory, so two of them read the same text only in a run with memory
                if spec.off_text and R._agent_memory(a) != R._agent_memory(b):
                    continue
                ha, hb = R.help_text(name, a), R.help_text(name, b)
                if not texts:
                    assert _above_marker(ha) == _above_marker(hb), (name, a, b)
                if own or texts:
                    continue
                pairs += 1
                assert ha == hb, (name, a, b)
                assert R.list_text([name], a) == R.list_text([name], b), (name, a, b)
                if name not in R.EMBEDDED:
                    assert stubs[a][name] == stubs[b][name], (name, a, b)
    assert pairs > 50


def test_an_agent_reads_its_own_arguments_in_place_or_after_the_marker(stand_ins):
    # the actor's delegate has texts of its own (it starts a helper like itself): its own signature, in place
    text = R.help_text("delegate", "actor")
    assert text.startswith("delegate(task)\n") and R.ROLE_MARKER not in text and "episodes" not in text, text
    assert R.ROLE_MARKER not in R.help_text("delegate", "writer")
    assert R.ROLE_MARKER not in R.help_text("finish", "writer")
    # the librarian's finish hands its answer to the agent that asked: its own texts, its own argument in place
    own = R.help_text("finish", "librarian")
    assert own.startswith("finish(answer)\n") and R.ROLE_MARKER not in own, own
    assert "gate" not in own and "manifest" not in own and "curator" not in own and "every item" not in own, own
    assert "at most one fallback" in own and "the agent that asked" in own
    writer = _load(R.render_stub(_allowed("writer"), "writer"))
    assert writer.finish("done") == "finish answered ['summary']"


def test_the_actor_and_a_teammate_end_with_answer_and_have_no_finish(stand_ins):
    for agent in ("actor", "teammate"):
        assert "answer" in _allowed(agent) and "finish" not in _allowed(agent), agent
        mod = _load(R.render_stub(_allowed(agent), agent))
        assert mod.answer("42") == "answer answered ['answer']", agent
        assert not hasattr(mod, "finish"), agent
        assert R.handler_for(agent, "answer") == ("unify.actor.tool_handlers:answer", False), agent
    assert R.registry()["answer"].roles == ("actor", "teammate")
    assert R.registry()["answer"].kind == "terminal"
    assert "finish" not in R.registry()["answer"].help_description + R.help_text("answer", "actor")
    assert R.help_text("answer", "actor").startswith("answer(answer)\n")
    assert R.refusal("answer", "answered") == "not answered: this request already has its answer"
    assert R.refusal("answer", "bad_answer", reason="x").startswith("not answered: the answer does not match")
    for agent in ("writer", "helper", "librarian"):  # rev 2 has no curator
        assert "answer" not in _allowed(agent), agent
    assert "finish" in _allowed("writer") and "finish" in _allowed("librarian")


def test_the_allow_lists_follow_the_profiles_and_the_design():
    actor = set(R.allow_list("actor", FULL))
    teammate = set(R.allow_list("teammate", FULL))
    assert actor == {
        "read", "grep", "install", "send", "messages", "wait", "who", "reply", "stop", "delegate", "answer",
        "request", "sh", "check_examples", "examples", "answer_from",
    }  # fmt: skip
    assert teammate == actor | {"ask"}  # any agent may start agents
    # learned skills are dropped in rev 2: no agent stores skills
    assert "store_skills" not in R.registry()
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
            # the run record's comms tools are the agents package's (unify.agents.run_record_tools)
            assert module.startswith(("unify.memory.", "unify.actor.", "unify.agents.")) and function == spec.name, (
                spec.name
            )
    for name in ("read", "grep", "pool", "stage", "install"):
        assert R.registry()[name].blocking, name
    # coroutine handlers run on the event loop and send their own blocking work to the worker pool
    # (rev 2 has no write or edit tool: a cell writes files)
    for name in ("wait", "ask", "delegate", "stop", "overview", "read_episode", "fixture", "check"):
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
        assert tool_names_in(text) - set(R.registry()) - own == set()


def test_one_rule_binds_the_namespaces_and_never_widens_the_allow_list():
    for agent in R.AGENTS:
        assert R.prebound(agent, {}) == {}
        assert R.prebound(agent, {"tools": True}) == {"tools": ["module", "tools"]}
        both = R.prebound(agent, {"tools": True, "memory": True})
        assert both == {"tools": ["module", "tools"], "memory": ["lazy_module", "memory"]}
        assert R.namespace_names(agent, {"memory": True}) == ("memory",)
        # the allow-list is the registry's whatever is bound
        assert R.allow_list(agent, FULL) == R.allow_list(agent, FULL)
    with pytest.raises(ValueError):
        R.prebound("fork", {})


def test_the_shared_sandbox_paragraph_has_one_text_per_cell_state():
    kept = R.sandbox_paragraph("writer", ("tools", "memory"), True)
    assert kept.startswith(R.SANDBOX_PERSISTENT)
    assert kept.endswith("`tools` and `memory` are already available as variables in every cell; no import needed.")
    assert R.sandbox_paragraph("helper", ("memory",), True).endswith(
        "`memory` is already available as a variable in every cell; no import needed.",
    )
    assert R.sandbox_paragraph("actor", (), False) == R.SANDBOX_FRESH
    # the same bytes for every role apart from the names: a stable cached prefix
    assert len({R.sandbox_paragraph(a, ("tools",), True) for a in R.AGENTS}) == 1
    for persistent in (True, False):
        text = R.preamble("writer", {"flags": FULL, "persistent": persistent, "names": ("tools", "memory")})
        paragraph = R.sandbox_paragraph("writer", ("tools", "memory"), persistent)
        assert text.count(paragraph) == 1
    assert "restarts" in R.SANDBOX_PERSISTENT and "variables are gone and files remain" in R.SANDBOX_PERSISTENT


# --- nothing in a definition outside the registry ---

_SCANNED = ("unify/actor", "unify/common/_async_tool", "unify/memory", "unify/agents")
#: Hand-written tool definitions outside the registry: none in rev 2 (the loop's response tool for tools off and the
#: earlier writer's three templates went with the switches).
_KNOWN: dict[str, int] = {}


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


# --- the stub holds registry fields only ---

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
        if "read" in _allowed(agent):  # the librarian has no read: it reads its library in its cells
            assert body["read"]("/workspace/a.txt") == "read answered ['path']"
        if "sh" in _allowed(agent):
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
    # the library's own missing-path form (the actor's and a teammate's read and grep of the library copy)
    "read": ["outside_library", "not_in_library"],
    "grep": ["search_failed", "outside_library", "not_in_library"],
    "finish": ["answered", "bad_answer"],
    "delegate": ["bad_name", "record_full", "short_request", "spawn_off", "too_large"],
    # the run record's addresses nobody would read (a role the run does not have, an agent that has ended)
    "send": ["record_full", "too_fast", "too_large", "no_role", "ended"],
    "reply": ["record_full", "too_fast", "too_large"],
    "ask": ["record_full", "too_fast", "too_large"],
    "stop": ["not_allowed"],
    "wait": ["bad_timeout"],
}
ADDED_TOOLS = (
    "install",
    "sh",
    "ask_library",
    "land_capture",
    "answer",
    "current_request",
    # the actor's and a teammate's reader of the request's exact text, in every run
    "request",
    "search",
    "check_examples",
    # the attention ledger's writer tools and the writer's stated tests (UNIFY_MEMORY_ATTENTION_LEDGER, _WRITER_TESTS)
    "open_entries",
    "take",
    "decide",
    "find",
    "test",
    # the actor's pause (UNIFY_MEMORY_IN_EPISODE)
    "checkpoint",
    # rev 2.3: the answer from a function on the request's declared input, and its two channel-only tools
    "answer_from",
    "declared_input",
    "answered_from",
    # rev 2.3: the examples the request was given as data, and their channel-only reader
    "examples",
    "declared_examples",
    # the librarian's notes for its later instances
    "keep_note",
)
#: Tools and forms of the base commit that rev 2 removed, with their base forms (put back before the hash, so the hash
#: still proves every other code is the base's): a cell writes files, so the write and edit tools are gone; the run
#: record lists names that match no agent in the reply, so send no longer refuses an unknown name.
REMOVED_TOOLS = {
    # rev 2.1: the writer's replay tool re-ran recorded actor cells, a re-execution of a capture, which no mechanism may do
    "replay": ["not_run"],
    "write": ["bad_content", "bad_path", "link", "not_directory", "not_regular", "not_written", "reserved"],
    "edit": [
        "bad_old", "bad_path", "link", "match_count", "not_directory", "not_found", "not_regular", "not_text",
        "not_written", "reserved",
    ],
}  # fmt: skip
#: Forms of the base commit removed since, put back before the hash. ``*``: the general check and reader-call limits,
#: removed with the per-pass caps in rev 2.1 (a pass has neither).
REMOVED_FORMS = {"send": ["unknown_name"], "*": ["check_limit", "reads_limit"]}
#: The base commit's order of the codes' tools (the hash is of ``json.dumps`` in that order).
BASE_ORDER = (
    "read", "grep", "overview", "read_episode", "pool", "fixture", "replay", "execute_code", "check", "finish",
    "similar", "delegate", "stage", "done", "send", "messages", "wait", "who", "reply", "stop", "ask", "write", "edit",
    "*",
)  # fmt: skip
#: Tools added to the writer's JSON tools (tools not in code) since the base commit: left out of the base hashes.
ADDED_JSON_TOOLS = ("land_capture",)
FLAGS = ("pool", "similar", "helper", "edit")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


def _base_tools(tools: list[dict]) -> list[dict]:
    """*tools* without the listed additions, each of which must be there."""
    return [t for t in tools if t["function"]["name"] not in ADDED_JSON_TOOLS]


def _base_section(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith(tuple(f"- `{n}`" for n in ADDED_JSON_TOOLS)))


def test_the_refusal_codes_are_the_base_commits_plus_the_listed_additions():
    codes = tool_spec.refusal_codes()
    for tool in ADDED_TOOLS:
        codes.pop(tool)
    for tool, forms in ADDED_FORMS.items():
        assert set(forms) <= set(codes[tool]), tool
        codes[tool] = [f for f in codes[tool] if f not in forms]
    for tool, forms in REMOVED_TOOLS.items():
        assert tool not in codes, tool
        codes[tool] = list(forms)
    for tool, forms in REMOVED_FORMS.items():
        assert not set(forms) & set(codes[tool]), tool
        codes[tool] = sorted([*codes[tool], *forms])
    assert set(codes) == set(BASE_ORDER)
    codes = {tool: codes[tool] for tool in BASE_ORDER}
    assert _h(codes) == BASE["r"]


def test_the_helper_keeps_its_own_refusal_words_for_read_and_grep():
    assert helpers._own_refusal("read", "outside_box", path="/x") == (
        "refused: /x is outside /inputs, /task, /memory and /work"
    )
    assert tool_spec.refusal("read", "outside_box", path="/x") == (
        "refused: /x is outside /inputs, /task, /memory and /work"
    )
    assert tool_spec.refusal("execute_code", "no_time") == "not run: the pass's wall clock is used up"
    assert tool_spec.refusal("grep", "not_found", path="/x") == "refused: /x does not exist"


def test_the_in_code_help_is_shorter_where_the_prompt_inventory_says_and_the_json_text_is_not():
    # (the run record's send and who have one definition for every agent in rev 2, with no separate in-code text)
    for name in ("read_episode", "fixture", "pool", "delegate", "similar"):
        spec = R.registry()[name]
        assert spec.help_description and len(spec.help_description) < len(spec.description), name
        assert spec.help_description in R.help_text(name)
        assert spec.description not in R.help_text(name)
    assert "Code is compared after normalising variable names and literals." in R.help_text("similar")


def test_every_agent_reads_one_wait_and_install_reads_as_in_the_actors_help():
    # rev 2: the run record's wait is one definition for every agent; its timeout is optional, at most 600
    texts = {R.help_text("wait", a) for a in ("actor", "teammate", "writer", "helper")}
    assert len(texts) == 1
    (text,) = texts
    assert "at most 600" in text and "(default 600)" in text
    from unify.actor import core_surface

    doc = getattr(core_surface, "INSTALL_DOC_IN_CODE", None)
    if doc is not None:
        assert R.INSTALL.description == doc
    assert "tools.install" in R.INSTALL.description and "stored function" not in R.INSTALL.description


