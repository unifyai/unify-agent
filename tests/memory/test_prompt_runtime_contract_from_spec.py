"""The prompt-to-runtime contract: every call a role's prompt shows runs, as written, in that role's own sandbox.

For each role (the actor, a teammate, the writer and a helper), with tools called from code, in library
mode, over a small fixture library (one package with a function, its test and a note) and one recorded episode
``e1``:

1. the role's real prompt is the first request the run sends to its (scripted) model: its system message and first
   user message, with no stand-in text;
2. every inline code span and fenced block is taken from it (``tests/memory/prompt_calls.py``); a span that does not
   parse goes to the reviewed list ``tests/memory/data/contract_skipped.txt``;
3. each placeholder is filled by one fixed rule (:data:`ANGLE_RULES`, :data:`BARE_RULES`); a call shown with
   ``...`` is the tool's first registry example; a placeholder with no rule fails the test;
4. the first cell reads back the names bound before any cell; the next runs every call, each in its own ``try``;
   the end call (``tools.answer``, ``tools.finish``, ``tools.done``) runs last;
5. a ``NameError``, ``ImportError``, an ``AttributeError`` on ``memory`` or ``tools``, a signature ``TypeError``,
   or an ``unknown_tool`` / "not available" refusal fails the test. ``tools.Refused`` otherwise passes: the call
   reached the host with a valid shape.

Also here: the reverse check (every pre-bound name is named in the prompt, and every name the prompt says is already
available is bound before the first cell), every role's first request opening with the shared opening (the
persistence and namespace sentences, no host path), and the episode-end fork naming no call it cannot make.

The memory roles run in their real boxes with the real handlers over the fixture pass (the harness of
``tests/memory/test_tool_channel_box.py``); the actor and a teammate run in the real sandboxed worker with
``InCodeTools`` (``tests/actor/code_act/test_in_code_tools_wiring.py``), where every tool handler the contract does
not need answers with fixed text. Keyless: the models are scripted.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import ENV_SECRET, TOKEN_VALUE, needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _text, code
from tests.memory import prompt_calls as pc
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import LIB
from tests.memory.test_library_helper import _commit
from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import _call
from tests.scripted_model import ScriptedModel, reply
from unify.actor import in_code_tools
from unify.common import tool_registry as R
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action, Cell
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.memory.library_checks import LibraryConfig
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.settings import SETTINGS

# ── assumed names: the integrator adapts these to the build in this one block ─────────────────────────────────
#: Design D.1, the persistent form of the shared paragraph (its opening words).
PERSISTENT_SENTENCE = "Your Python sandbox is persistent and stateful"
#: Design D.1, the names sentence: "{names} are already available as variables in every cell; no import needed."
NAMES_SENTENCE_END = "already available as variables in every cell; no import needed."
#: The shared opening's first line and the start of its last part (``tool_registry.preamble``): how a memory agent
#: ends its work, or, for the actor and a teammate (whose role says how they answer), the line about the record.
OPENING_HEADING = "## How you work here"
OPENING_LAST = "You end your work with"
OPENING_LAST_ACTOR = "Other agents may share a record with you;"
#: Design B.1: ``tool_registry.prebound_names(agent, facts)``; *facts* as this run's configuration gives them. The
#: keys it reads: ``tools`` (tools in code), ``memory`` (the library package is in the box) and, for the actor,
#: ``clarification`` (its session can ask its requester; left off here, so ``request_clarification`` is not required).
PREBOUND_FACTS = {
    "actor": {"tools": True, "memory": True, "clarification": False},
    "teammate": {"tools": True, "memory": True},
    "writer": {"tools": True, "memory": True},
    "helper": {"tools": True, "memory": True},
}


def prebound_names(role: str) -> list[str]:
    return list(R.prebound_names(role, PREBOUND_FACTS[role]))


#: The reviewed list of spans that do not parse, one ``<role>\t<span>`` per line (design F.1, step 2).
SKIPPED_LIST = Path(__file__).parent / "data" / "contract_skipped.txt"
#: An agent's run-record name, made per run (``actor-`` and a hex suffix).
_RUN_NAME = re.compile(r"\bactor-[0-9a-f]{4,}\b")
#: The spans a prompt shows that parse but are not calls to run (what not to call, a form written in library code, a
#: command quoted for reference), each with its reason: the reviewed list of the other contract test, kept in one place.
from tests.memory.test_prompt_runtime_contract import NOT_RUN  # noqa: E402
# ── end of assumed names ──────────────────────────────────────────────────────────────────────────────────────

ROLES = ("actor", "teammate", "writer", "helper")
FUNCTION_ID = "memory.text.dates:parse_date"
PACKAGE = "text"
MODULE = "dates"
NOTE_PATH = "notes/text/dates.md"
TEST_ID = "memory/text/tests/test_dates.py::test_parse_date"
#: the fixture function's first recorded input (its test's call)
FIRST_INPUT = "2026-10-09"
REQUEST_TEXT = "Read the date in this note and report it: 2026-10-09."

#: ``<placeholder>`` -> its text (quoted where the span does not quote it).
ANGLE_RULES = {
    "id": FUNCTION_ID,
    "item": FUNCTION_ID,
    "name": PACKAGE,
    "package": PACKAGE,
    "section": PACKAGE,
    "module": MODULE,
    "episode": "e1",
    "path": NOTE_PATH,
    "test id": TEST_ID,
    "part": "request",
}
#: A bare placeholder name -> the Python expression it stands for (bound before the span runs).
BARE_RULES = {
    "value": repr(FIRST_INPUT),
    "text": repr(REQUEST_TEXT),
    "request": repr(REQUEST_TEXT),
    "episode": "'e1'",
    "episodes": "['e1']",
    "parts": "['request']",
    "part": "'request'",
    "path": repr(NOTE_PATH),
    # help(name) documents a Python object; tools.help(name) one of the role's tools
    "name": "'print'",
    "tools.help:name": "'read'",
    "obj": "tools",
    "command": "'ls'",
    "item": repr(FUNCTION_ID),
    "target": repr(FUNCTION_ID),
    "package": repr(PACKAGE),
    "capture_id": "'0000000000000000'",
    "summary": "'contract check'",
    "claims": "[]",
    "answer": "'contract check'",
    "task": "'contract check'",
    "pattern": "'date'",
    "code": "'print(1)'",
    "cell": "0",
    "dest": "'memory/text/tests/data/contract.json'",
    "source": "'request:0'",
    "agent": "'writer'",
    "factory": "None",
    # the helper's prompt shows ``tools.ask(question)``; ``tools.stage``'s first example stages ``src``
    "question": "'@writer is there anything to add?'",
    "src": "'def area(w, h):\\n    return w * h\\n'",
}

MEMORY_END = {
    "writer": "tools.finish('contract check done')\n",
    "helper": "tools.done('contract check done', [])\n",
}


def _example(tool: str) -> str | None:
    spec = R.registry().get(tool)
    return spec.examples[0] if spec is not None and spec.examples else None


def _prepare(text: str, names: list[str] | None) -> pc.Plan:
    return pc.prepare(
        text,
        angle_rules=ANGLE_RULES,
        bare_rules=BARE_RULES,
        namespace=names or [],
        example=_example,
        skip=NOT_RUN,
    )


def _prompt(messages: list[dict]) -> str:
    """The role's prompt as its first request carries it: the system message(s) and the first user message."""
    system = [_text(m.get("content")) for m in messages if m.get("role") == "system"]
    user = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    return "\n\n".join([*system, user])


def _system(messages: list[dict]) -> str:
    return next((_text(m.get("content")) for m in messages if m.get("role") == "system"), "")


# ── the memory roles: a scripted model whose cells are made from the conversation so far ───────────────────────


class Steps:
    """A memory agent's model turn (``model_turn`` / a helper's turn): each turn's cell comes from ``plan(self, n)``
    (n counts from 1; None: the role's end call); it keeps every request it was sent and every cell output."""

    def __init__(self, plan, end: str) -> None:
        self.plan, self.end = plan, end
        self.requests: list[list[dict]] = []
        self.outputs: dict[str, str] = {}

    async def __call__(self, messages, tools):
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = _text(m.get("content"))
        n = len(self.requests)
        cell = self.plan(self, n)
        return _call(f"c{n}", "execute_code", {"code": cell if cell is not None else self.end}), "0.001"


def contract_plan(state: dict):
    """Cell 1 reads the namespace; cell 2 runs every non-end call of the first request's prompt; cell 3 the end
    calls."""

    def plan(steps: Steps, n: int):
        if n == 1:
            state["prompt"] = _prompt(steps.requests[0])
            return pc.NAMESPACE_CELL
        if n == 2:
            state["namespace"] = pc.namespace(steps.outputs.get("c1", ""))
            state["plan"] = _prepare(state["prompt"], state["namespace"])
            return pc.contract_cell([c for c in state["plan"].calls if not c.end])
        if n == 3:
            ends = [c for c in state["plan"].calls if c.end]
            return pc.contract_cell(ends) if ends else None
        return None

    return plan


def probe_plan(state: dict):
    """Cell 1 reads the namespace; then the role ends."""

    def plan(steps: Steps, n: int):
        if n == 1:
            state["first"] = steps.requests[0]
            state["prompt"] = _prompt(steps.requests[0])
            return pc.NAMESPACE_CELL
        if n == 2:
            state["namespace"] = pc.namespace(steps.outputs.get("c1", ""))
        return None

    return plan


def _episode():
    return _ep(
        episode_id="e1",
        request=[REQUEST_TEXT],
        transcript=[{"role": "user", "content": REQUEST_TEXT}],
        cells=[Cell(0, "print(parse_date('2026-10-09'))", "2026-10-09")],
        actions=[Action(0, "notes", "read", [], {}, {"date": FIRST_INPUT}, "ok", "read")],
    )


DELEGATE_CELL = "print(tools.delegate('Run the calls you are given in your own box.', episodes=['e1']))\n"
WAIT_CELL = "for _ in range(12):\n    print(tools.wait()[:200])\n"


def _memory_pass(tmp_path: Path, role: str, model: Steps, helper: Steps | None = None):
    """One pass of *role* (the writer; a helper is started by a scripted writer) over the fixture library and
    episode, tools called from code, in real boxes."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    _commit(mem, LIB, "seed the library")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _episode()
    ev.index_episode(ep, "1" * 40)
    gate = Gate(
        mem,
        ev,
        BlobStore(tmp_path / "b"),
        action_lookup=lambda eid, i: ep.actions[i] if (eid, i) == ("e1", 0) else None,
        library_mode=LibraryConfig(episodes={"e1": ep}.get),
    )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), 
            pass_record_root=str(tmp_path / "out"),
            max_calls=8,
            cell_timeout_s=300.0,
            deadline_s=1500.0,
        ),
    )
    if helper is not None:
        writer.helper_turn = lambda label=None: helper
    req = PassRequest("incremental", PACKAGE, ["e1"], False)
    out = asyncio.run(writer.run(req, "p1"))
    return out, writer


def _delegating_writer() -> Steps:
    def plan(steps: Steps, n: int):
        return {1: DELEGATE_CELL, 2: WAIT_CELL}.get(n)

    return Steps(plan, MEMORY_END["writer"])


def _helper_waits_answer_at_once(monkeypatch) -> None:
    """A helper's ``wait`` and ``ask`` answer at once (as ``no_waiting`` in the other contract test): the scripted
    writer only waits, so a helper's real ``wait`` or ``ask`` would wait for it until the pass's wall clock. The
    writer's own ``wait`` stays real, so its wait cell sees the helper end. ``wait`` is the run record's (the registry
    routes it to :mod:`unify.agents.run_record_tools` for every agent); ``ask`` is memory's."""
    from unify.agents import run_record_tools as RT
    from unify.memory import tool_handlers as TH

    for module, name, text in ((RT, "wait", "no new messages"), (TH, "ask", "no answer yet")):
        real = getattr(module, name)

        async def handler(ctx, args, _real=real, _text=text):
            if isinstance(getattr(ctx, "state", None), TH.HelperCalls):
                return _text
            return await _real(ctx, args)

        monkeypatch.setattr(module, name, handler)


def _run_memory_role(tmp_path: Path, role: str, plan_of) -> tuple[dict, Steps, Any]:
    state: dict = {}
    steps = Steps(plan_of(state), MEMORY_END[role])
    if role == "helper":
        out, _ = _memory_pass(tmp_path, "writer", _delegating_writer(), helper=steps)
    else:
        out, _ = _memory_pass(tmp_path, role, steps)
    return state, steps, out


# ── the actor and a teammate ─────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def actor_world(core_world, monkeypatch):  # noqa: F811
    """Tools in code on, a library-mode request over the fixture library (``memory`` bound in the cells), and every
    in-code session the test opens."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    paths = Paths.under(core_world["state"])
    mem = Repo.init_bare(paths.memory)
    _commit(mem, LIB, "seed the library")
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = mem.head()
    run._shape_rows = lambda consolidate, sha=None: {}  # no evidence store here
    run._open_library(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    yield {"world": core_world, "opened": collect.opened, "run": run, "paths": paths}


def _fixed_handlers(monkeypatch, keep_for_actor: frozenset = frozenset()) -> None:
    """Every actor and teammate tool handler but ``answer`` answers with fixed text; those in *keep_for_actor* stay
    real for the actor itself (the teammate test needs the actor's ``delegate`` and ``wait``)."""
    from unify.actor import tool_handlers as H

    for name in R.registry():
        where, blocking = R.handler_for("actor", name)
        if not where.startswith("unify.actor.tool_handlers:") or name == "answer":
            continue
        attr = where.split(":", 1)[1]
        real = getattr(H, attr, None)
        if real is None:
            continue
        monkeypatch.setattr(H, attr, _fixed(name, real, blocking, keep_for_actor))


def _fixed(name: str, real, blocking: bool, keep: frozenset):
    text = f"(a fixed reply from {name})"

    if blocking:

        def handler(ctx, args):
            if name in keep and getattr(ctx, "agent", None) == "actor":
                return real(ctx, args)
            return text

        return handler

    async def handler(ctx, args):  # noqa: F811
        if name in keep and getattr(ctx, "agent", None) == "actor":
            return await real(ctx, args)
        return text

    return handler


def _last_tool_text(call) -> str:
    tools = [_text(m.get("content")) for m in call.messages if m.get("role") == "tool"]
    return tools[-1] if tools else ""


def _scripted_contract(state: dict, end_reply: str) -> list:
    """The actor's (or a teammate's) turns for the contract: namespace, the calls, the end calls, then a reply."""
    state.setdefault("outputs", [])

    def probe(call):
        state["first"] = call.messages
        state["prompt"] = _prompt(call.messages)
        return reply(calls=[code(pc.NAMESPACE_CELL)])

    def calls(call):
        state["namespace"] = pc.namespace(_last_tool_text(call))
        state["plan"] = _prepare(state["prompt"], state["namespace"])
        return reply(calls=[code(pc.contract_cell([c for c in state["plan"].calls if not c.end]))])

    def ends(call):
        state["outputs"].append(_last_tool_text(call))
        last = [c for c in state["plan"].calls if c.end]
        return reply(calls=[code(pc.contract_cell(last))]) if last else reply(end_reply)

    def final(call):
        state["outputs"].append(_last_tool_text(call))
        return reply(end_reply)

    return [probe, calls, ends, final]


def _scripted_probe(state: dict, end: Any) -> list:
    def probe(call):
        state["first"] = call.messages
        state["prompt"] = _prompt(call.messages)
        return reply(calls=[code(pc.NAMESPACE_CELL)])

    def after(call):
        state["namespace"] = pc.namespace(_last_tool_text(call))
        return end if not callable(end) else end(call)

    return [probe, after]


TEAMMATE_START = "print(tools.delegate('Run the calls you are given in your own sandbox.'))\n"
TEAMMATE_WAIT = (
    "got = ''\n"
    "for _ in range(8):\n"
    "    got += tools.wait(timeout=240)\n"
    "    if '(done)' in got:\n"
    "        break\n"
    "print(got[-1500:])\n"
)


def _run_actor(actor_world, monkeypatch, role: str, turns_of) -> tuple[dict, ScriptedModel]:
    state: dict = {}
    if role == "actor":
        _fixed_handlers(monkeypatch)
        model = ScriptedModel(actor=turns_of(state, "done"), allow=("query_llm",))
    else:
        _fixed_handlers(monkeypatch, frozenset({"delegate", "wait"}))
        model = ScriptedModel(
            actor=[reply(calls=[code(TEAMMATE_START)]), reply(calls=[code(TEAMMATE_WAIT)]), reply("done")],
            helper=turns_of(state, "the calls ran"),
            allow=("query_llm",),
        )
    asyncio.run(asyncio.wait_for(_act(model, REQUEST_TEXT), 900))
    return state, model


def _run_role(role: str, tmp_path, request, monkeypatch, *, contract: bool) -> dict:
    """Run *role* once: the contract (every call) or a probe (namespace, then the end). Returns the state: the first
    request's messages (``first``), its prompt text, the namespace, the plan and the cell outputs."""
    if role in ("actor", "teammate"):
        actor_world = request.getfixturevalue("actor_world")
        if contract:
            state, model = _run_actor(actor_world, monkeypatch, role, _scripted_contract)
        else:
            end_reply = reply("done") if role == "actor" else reply(calls=[code("tools.answer('ok')\n")])
            state, model = _run_actor(
                actor_world,
                monkeypatch,
                role,
                lambda st, _end: _scripted_probe(st, end_reply),
            )
        state["world"] = actor_world
        state["sent"] = [c.request for c in model.calls]
        return state
    # the memory agents' briefs and boxes follow the switch, as in a run (PassConfig.tools_in_code alone is not it)
    if role == "helper":
        _helper_waits_answer_at_once(monkeypatch)
    if contract:
        state, steps, out = _run_memory_role(tmp_path, role, contract_plan)
        state["outputs"] = [steps.outputs.get("c2", ""), steps.outputs.get("c3")]
    else:
        state, steps, out = _run_memory_role(tmp_path, role, probe_plan)
    state["out"] = out
    state["sent"] = steps.requests
    state["first"] = steps.requests[0] if steps.requests else []
    return state


# ── 1. every call a prompt shows runs as written ─────────────────────────────────────────────────────────────


@needs_box
@needs_bwrap
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("role", ROLES)
def test_every_call_the_role_prompt_shows_runs_as_written_in_its_sandbox(role, tmp_path, request, monkeypatch):
    state = _run_role(role, tmp_path, request, monkeypatch, contract=True)
    assert state.get("prompt"), f"{role}: no request reached the model"
    assert state.get("namespace") is not None, f"{role}: the namespace cell gave no names: {state.get('outputs')}"
    plan: pc.Plan = state["plan"]
    assert plan.calls, f"{role}: its prompt shows no call at all"
    got: dict[int, dict] = {}
    texts = [t for t in state.get("outputs", []) if t]
    for text in texts:
        got.update(pc.results(text))
    ended = len(texts) < 2  # the end call's cell output never comes back when it ended the work
    bad = plan.no_rule + pc.failures(plan.calls, got, ended=ended)
    assert not bad, f"{role}: calls its prompt shows that do not run as written:\n" + "\n".join(bad)


# ── 2. the reverse check ─────────────────────────────────────────────────────────────────────────────────────


@needs_box
@needs_bwrap
@pytest.mark.timeout(900)
@pytest.mark.parametrize("role", ROLES)
def test_every_prebound_name_is_named_and_every_promised_name_is_bound(role, tmp_path, request, monkeypatch):
    state = _run_role(role, tmp_path, request, monkeypatch, contract=False)
    bound = set(state.get("namespace") or [])
    assert bound, f"{role}: the namespace cell gave no names"
    prompt = state["prompt"]
    names = prebound_names(role)
    assert {"tools", "memory"} <= set(names), names
    # an exception class the harness raises into cells (``ModelRequestError``) is bound for ``except`` clauses, not
    # offered as something to call, so only the names a cell uses need the prompt to name them; all must be bound
    # Python's own built-in names (``help``) need no introduction either: the box binds its own version of them
    import builtins as _builtins

    missing = [
        n
        for n in names
        if not n[:1].isupper() and not hasattr(_builtins, n) and f"`{n}`" not in prompt and f"`{n}(" not in prompt
    ]
    assert not missing, f"{role}: pre-bound but never named in its prompt: {missing}"
    promised = pc.opening_names(prompt, NAMES_SENTENCE_END)
    assert promised, f"{role}: its prompt has no names sentence"
    assert set(promised) <= bound, f"{role}: promised as already available but unbound: {set(promised) - bound}"
    assert set(names) <= bound, f"{role}: pre-bound names missing from the cell: {set(names) - bound}"
    # memory is bound lazily: its name is there before any cell imports the library
    assert "memory" in bound


# ── 3. every role's first request opens with the shared opening ─────────────────────────────────────────────


def _host_paths(state: dict, tmp_path: Path) -> list[str]:
    paths = {str(tmp_path), os.path.realpath(tmp_path), os.path.expanduser("~")}
    world = state.get("world")
    if world:
        for root in ("home", "state", "workspace"):
            if root in world["world"]:
                paths |= {str(world["world"][root]), os.path.realpath(world["world"][root])}
    return sorted(p for p in paths if p and p != "/")


@needs_box
@needs_bwrap
@pytest.mark.timeout(900)
@pytest.mark.parametrize("role", ROLES)
def test_each_role_first_request_carries_the_shared_opening_with_persistence_and_names(
    role,
    tmp_path,
    request,
    monkeypatch,
):
    state = _run_role(role, tmp_path, request, monkeypatch, contract=False)
    first = state["first"]
    system = _system(first)
    assert system.startswith(OPENING_HEADING + "\n"), f"{role}: {system[:400]!r}"
    whole = json.dumps(first)
    # the persistence sentence, once, inside the opening (before its closing part)
    assert PERSISTENT_SENTENCE in system, f"{role}: no persistence sentence in {system[:1500]!r}"
    assert whole.count(PERSISTENT_SENTENCE) == 1, f"{role}: the persistence sentence is sent more than once"
    if role in ("actor", "teammate"):
        # rev 2: the actor's and a teammate's opening ends with no record line and no end line (they answer by their
        # role's rule); the run record's common section opens the first message instead, once
        assert OPENING_LAST not in system and OPENING_LAST_ACTOR not in system, role
        record = R.COMMON_SECTION.splitlines()[0]
        users = [_text(m.get("content")) for m in first if m.get("role") == "user"]
        assert users and record in users[0] and record not in system and whole.count(record) == 1, role
        last = NAMES_SENTENCE_END
        assert system.index(PERSISTENT_SENTENCE) < system.index(last), role
    else:
        last = OPENING_LAST
        assert last in system and system.index(PERSISTENT_SENTENCE) < system.index(last), role
    assert "fresh interpreter" not in system and "Only files persist between calls" not in system, role
    # the namespace sentence names tools and memory, once, inside the opening
    names = pc.opening_names(system, NAMES_SENTENCE_END)
    assert {"tools", "memory"} <= set(names), f"{role}: names sentence lists {names}"
    assert whole.count(NAMES_SENTENCE_END) == 1, f"{role}: the names sentence is sent more than once"
    assert system.index(NAMES_SENTENCE_END) <= system.index(last), role
    # nothing tells the agent to import what is already bound
    assert "`import memory`" not in whole, f"{role}: still tells the agent to import memory"
    # box paths only: no host path, no planted secret, in anything the role is sent
    for path in _host_paths(state, tmp_path):
        assert path not in whole, f"{role}: a host path reached the request: {path}"
    for value in (ENV_SECRET, TOKEN_VALUE):
        assert value not in whole, role


@needs_box
@needs_bwrap
@pytest.mark.timeout(900)
@pytest.mark.parametrize("role", ROLES)
def test_spans_that_do_not_parse_are_on_the_reviewed_list(role, tmp_path, request, monkeypatch):
    state = _run_role(role, tmp_path, request, monkeypatch, contract=False)
    # an agent's name on the run record is made per run (``actor-<hex>``): whether it parses depends on its suffix
    # (``actor-0356`` does not, ``actor-abcd`` does), and it is never a call, so it is left out
    seen = sorted({s for s in _prepare(state["prompt"], state.get("namespace")).unparsed if not _RUN_NAME.fullmatch(s)})
    listing = "\n".join(f"{role}\t{s}" for s in seen)
    assert SKIPPED_LIST.is_file(), f"no reviewed list at {SKIPPED_LIST}; this role's lines:\n{listing}"
    reviewed = sorted(
        {
            line.split("\t", 1)[1]
            for line in SKIPPED_LIST.read_text().splitlines()
            if line.startswith(role + "\t")
        },
    )
    assert seen == reviewed, f"{role}: unreviewed or stale entries; the role's lines now:\n{listing}"


# ── 4. the episode-end fork ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("in_code", ["on"])
def test_the_fork_instruction_names_no_call_it_cannot_make(in_code, monkeypatch):
    """The fork has no sandbox: its own instruction shows no call of ``tools``, ``memory`` or a tool by its bare name
    beyond the run record's, which it may make (rev 2: every agent is on the record). (The functions it is asked to
    write, such as ``read_input(text)``, are not calls it makes.)"""
    from unify.memory.integration import fork

    text = fork.fork_instruction()
    assert text and fork.RECORD_CALLS_ONLY in text
    shown = pc.harness_calls(text, R.registry())
    assert shown == [], shown
