"""The prompt-to-runtime contract: every call a prompt shows runs, as written, in a real sandbox of that role.

For each role (the actor, a teammate, the writer, the curator and a helper; with ``UNIFY_TOOLS_IN_CODE`` on and,
for the memory agents, off) the real prompt is rendered by the role's own builders and taken from the request the
harness sends: the system message, the first message and the model-API tool texts. Every inline code span and
fenced block that parses as Python and calls something, or reads an attribute of a name, is a shown call. Each one
has its placeholders filled by one fixed rule (:data:`PLACEHOLDERS`, :data:`NAMES`, :data:`ELLIPSIS`) from a
non-empty fixture library (the layout tests' library: package ``text``, function ``memory.text.dates:parse_date``
with its test, one note) and episode ``e1``, and is then run, in prompt order, as one cell of the role's own
interpreter (the persistent sandbox the prompt describes). A cell first lists the namespace, so every name the
prompt says is already available is checked to be bound before any cell binds it.

A shown call fails the contract when it raises ``NameError``, ``ImportError`` (or ``ModuleNotFoundError``),
``TypeError`` (a signature the prompt shows that the runtime does not take), an ``AttributeError`` on ``memory`` or
``tools``, or is refused as an unknown or unavailable tool. A free name with no binding and no rule is a
placeholder with no rule, and fails as a ``NameError``. ``tools.Refused`` and ``tools.NotRun`` pass: the call
reached the host with a valid shape. Calls that end the agent's work (``finish``, ``done``) run last. The few
spans a prompt shows as something not to do, or as a pattern rather than a call, are listed in :data:`NOT_RUN`
with the reason, and each must still be in some prompt, so the list is reviewed whenever a prompt changes.

The episode-end fork has no sandbox: its instruction is checked to name no pre-bound name.

Keyless: the models are scripted and the cells run in the real boxes (bubblewrap and prlimit required). Calls that
would wait on another agent (``wait``, ``ask``) are answered at once by stand-in handlers, as the tool channel box
test does; for the actor, ``install`` and the record calls that start or stop agents answer fixed text.
"""

from __future__ import annotations

import ast
import builtins
import asyncio
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.memory.test_layout import LIB
from tests.memory.test_library_helper import _commit
from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import _ME, _call, _e1
from unify.common import tool_registry as R
from unify.settings import SETTINGS

PD = "memory.text.dates:parse_date"

#: Placeholders inside a shown string, each with its one value from the fixture library and episode.
PLACEHOLDERS = {
    "<id>": PD,
    "<item>": PD,
    "<name>": "text",
    "<package>": "text",
    "<episode>": "e1",
    "<part>": "request",
}
#: Free names a shown call uses as placeholders (bound neither by the sandbox nor by an earlier shown call), each
#: with the expression that fills it. ``<path>`` and ``path`` are the role's path to the fixture note.
NAMES = {
    "value": "'2026-10-09'",
    "summary": "'contract test summary'",
    "answer": "'contract test answer'",
    "claims": "[]",
    "question": "'@writer is there anything to add?'",
    "text": "'contract test message'",
    "request": "'contract test request'",
    "episode": "'e1'",
    "episodes": "['e1']",
    "parts": "['request']",
    "cell": "0",
    "code": "'x = 1'",
    "src": "'def area(w, h):\\n    return w * h\\n'",
    "command": "'true'",
    "name": "'read'",
    "obj": "len",
    "factory": "(lambda: __import__('asyncio').sleep(0, result='ran'))",
    "id": f"{PD!r}",
    "item": f"{PD!r}",
    "package": "'text'",
}
#: Calls written with ``...`` that are not registered tools: the call each stands for.
ELLIPSIS = {
    "query_llm": "query_llm('Reply with the word ok.')",
    "list_llms": "list_llms()",
}
#: Whole spans whose placeholder depends on the call: ``help(name)`` documents a bound object.
SPANS = {
    "help(name)": "help(len)",
}
#: Shown spans that are not calls to run, with the reason; each must still occur in some rendered prompt.
NOT_RUN: dict[str, str] = {
    "asyncio.run(...)": "shown as what not to call: the runtime owns the event loop",
    'MemoryInputError("<what was expected>")': "the form a library function raises, written in library code",
    (
        'subprocess.run([sys.executable, "-m", "pytest", "memory", "-q", "-p", "no:cacheprovider", "--rootdir", '
        '"/memory"], cwd="/memory", env={"PYTHONPATH": "/memory", "PYTEST_ADDOPTS": "-c /dev/null '
        '--import-mode=importlib", "PYTHONDONTWRITEBYTECODE": "1"})'
    ): "the gate's own test command, quoted for reference (a cell that runs it imports subprocess and sys)",
}
#: Exceptions a shown call must never raise.
FAILING = {"NameError", "ImportError", "ModuleNotFoundError", "TypeError"}
#: Calls that end the agent's work: run last.
TERMINAL = {"finish", "done"}


# --- extraction and filling (no box) --------------------------------------------------------------------------


_FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)
_SPAN = re.compile(r"`([^`\n]+)`")


def shown_code(text: str) -> list[str]:
    """Every fenced block and inline code span of *text*, in order (blocks first)."""
    blocks = _FENCE.findall(text)
    return [*blocks, *_SPAN.findall(_FENCE.sub("", text))]


def _root(node: ast.AST) -> Optional[str]:
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def is_shown_call(code: str, bound: frozenset[str] = frozenset({"tools", "memory"})) -> bool:
    """*code* parses and calls something, or reads an attribute of a name in *bound* (an attribute of any other
    name is a file name or a dotted id in prose, such as ``notes.md`` or ``channel.method``). An annotation alone
    (``memory.text.dates:parse_date``, an item id) is not a call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if all(isinstance(n, ast.AnnAssign) and n.value is None for n in tree.body):
        return False
    return any(
        isinstance(n, ast.Call) or (isinstance(n, ast.Attribute) and _root(n) in bound) for n in ast.walk(tree)
    )


def called_tool(code: str) -> Optional[str]:
    """The tool a span calls as ``tools.<name>(...)`` first, or the bare name a span calls, if any."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tools":
                return f.attr
            if isinstance(f, ast.Name):
                return f.id
    return None


class _Fill(ast.NodeTransformer):
    def __init__(self, bound: set[str], path: str) -> None:
        self.bound, self.path = bound, path

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str):
            text = node.value
            for key, value in {**PLACEHOLDERS, "<path>": self.path}.items():
                text = text.replace(key, value)
            return ast.copy_location(ast.Constant(text), node)
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound:
            if node.id == "path":
                return ast.copy_location(ast.Constant(self.path), node)
            if node.id in NAMES:
                return ast.copy_location(ast.parse(NAMES[node.id], mode="eval").body, node)
        return node


def _ellipsis_call(code: str) -> Optional[str]:
    """For a span that is one call written ``f(...)`` (or ``f(..., key=value)``): what it stands for (the tool's
    first registry example, with the span's keywords added, or :data:`ELLIPSIS`), else None."""
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    call = tree.body[0].value
    if not (isinstance(call, ast.Call) and len(call.args) == 1):
        return None
    arg = call.args[0]
    if not (isinstance(arg, ast.Constant) and arg.value is Ellipsis):
        return None
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tools":
        spec = R.registry().get(f.attr)
        if spec is None or not spec.examples:
            return None
        example = ast.parse(spec.examples[0])
        for node in ast.walk(example):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == f.attr:
                node.keywords = [*node.keywords, *call.keywords]
                return ast.unparse(example)
        return None
    if isinstance(f, ast.Name) and f.id in ELLIPSIS and not call.keywords:
        return ELLIPSIS[f.id]
    return None


def fill(code: str, bound: set[str], path: str) -> str:
    """*code* with its placeholders filled by the fixed rules; names in *bound* (the sandbox's and those earlier
    shown calls bound) are left as they are."""
    code = SPANS.get(code) or _ellipsis_call(code) or code
    tree = _Fill(bound, path).visit(ast.parse(code))
    return ast.unparse(ast.fix_missing_locations(tree))


def binds(code: str) -> set[str]:
    """The names a span binds at its top level (assignments, imports, definitions)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
    return out


def said_bound(text: str) -> set[str]:
    """The names a prompt says are already available, from its names sentences."""
    out: set[str] = set()
    for m in re.finditer(r"((?:`\w+`(?:, | and )?)+) (?:is|are) already available", text):
        out |= set(re.findall(r"`(\w+)`", m.group(1)))
    for m in re.finditer(r"`(\w+)` is already bound", text):
        out.add(m.group(1))
    for m in re.finditer(r"In every cell `(\w+)` is already bound", text):
        out.add(m.group(1))
    return out


def plan(texts: list[str], namespace: set[str], path: str, skip_tools: frozenset[str] = frozenset()):
    """The cells to run for *texts*, in prompt order: (span, filled code, terminal), without duplicates. Spans that
    call one of *skip_tools* (model-API tools, not Python) or are in :data:`NOT_RUN` are left out."""
    seen: set[str] = set()
    body, last = [], []
    bound = set(namespace)
    for text in texts:
        for span in shown_code(text):
            span = span.strip()
            if span in seen or not is_shown_call(span, frozenset(bound)):
                continue
            seen.add(span)
            if span in NOT_RUN:
                continue
            tool = called_tool(span)
            if tool in skip_tools and not span.startswith(("tools.", "memory.", "print(")):
                continue
            code = fill(span, bound, path)
            bound |= binds(code)
            (last if tool in TERMINAL and span.startswith("tools.") else body).append((span, code))
    return body, last[:1]


#: The actor's cell that starts the teammate under test.
START_TEAMMATE = "print(tools.delegate('Run every call your instructions show.'))"
#: The cell that lists the namespace before any shown call runs.
LIST_NAMES = "print('NAMES', sorted(k for k in globals() if not k.startswith('_')))"


def wrapped(i: int, code: str) -> str:
    """*code* as written, inside a try that reports how it ended on one line the test reads."""
    return (
        "try:\n"
        + textwrap.indent(code, "    ")
        + f"\n    print('CONTRACT {i} ok')\n"
        "except Exception as exc:\n"
        f"    print('CONTRACT {i} raised', type(exc).__name__, getattr(exc, 'form', None), repr(str(exc))[:400])\n"
    )


def verdicts(output: str) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for m in re.finditer(r"^CONTRACT (\d+) (ok|raised)(.*)$", output, re.M):
        out[int(m.group(1))] = (m.group(2), m.group(3).strip())
    return out


def failed(how: tuple[str, str] | None) -> Optional[str]:
    """Why a shown call's outcome fails the contract, or None."""
    if how is None:
        return "it never reported (the cell did not run to its end)"
    kind, rest = how
    if kind == "ok":
        return None
    exc = rest.split(" ", 1)[0]
    if exc in FAILING:
        return rest
    if exc == "AttributeError" and re.search(r"'(memory|tools)'", rest):
        return rest
    if "unknown_tool" in rest or "not available" in rest or "not_available" in rest:
        return rest
    return None


def names_listed(output: str) -> set[str]:
    m = re.search(r"^NAMES (\[.*\])$", output, re.M)
    assert m, output[-2000:]
    return set(ast.literal_eval(m.group(1)))


# --- the extraction rules themselves ----------------------------------------------------------------------------


def test_the_rules_fill_every_placeholder_and_keep_what_is_bound():
    assert shown_code("a `memory.show(\"<id>\")` b\n```python\nx = tools.read(path)\n```\n") == [
        "x = tools.read(path)\n",
        'memory.show("<id>")',
    ]
    assert is_shown_call("memory.index()") and is_shown_call("tools.Refused")
    assert not is_shown_call("memory.text.dates:parse_date") and not is_shown_call("MemoryInputError")
    assert not is_shown_call("notes.md") and not is_shown_call("channel.method")
    assert fill("tools.fixture(..., append=True)", {"tools"}, "/p") == (
        "tools.fixture('e1', 'memory/pkg/tests/data/page.html', source='3', append=True)"
    )
    assert fill('memory.show("<id>")', {"memory"}, "/p") == f"memory.show({PD!r})"
    assert fill("tools.read(path)", {"tools"}, "/memory/notes/text/dates.md") == (
        "tools.read('/memory/notes/text/dates.md')"
    )
    assert fill("tools.check(...)", {"tools"}, "/p") == "print(tools.check())"
    assert fill("memory.find(value)", {"memory", "value"}, "/p") == "memory.find(value)"
    assert fill("help(name)", set(), "/p") == "help(len)"
    assert fill("tools.help(name)", {"tools"}, "/p") == "tools.help('read')"
    assert said_bound("`tools` and `memory` are already available as variables") == {"tools", "memory"}
    assert failed(("raised", "NameError None \"name 'x' is not defined\"")) is not None
    assert failed(("raised", "Refused outside_roots 'refused'")) is None
    assert failed(("raised", "AttributeError None \"module 'tools' has no attribute 'nope'\"")) is not None


# --- the memory agents: a scripted pass in the real boxes ---------------------------------------------------------


class Contract:
    """A scripted model turn for a memory agent: on its first call it reads the prompt it was sent, plans the
    cells (:func:`plan`), then sends them one by one and keeps every cell's output. *json_end*: with tools in code
    off, the model-API call that ends the work (name, arguments)."""

    def __init__(
        self,
        role: str,
        path: str,
        *,
        in_code: bool,
        json_end: tuple[str, dict],
        extra_texts: tuple[str, ...] = (),
        before_end: Any = None,
        first_cells: tuple[str, ...] = (),
        texts_override: Optional[list[str]] = None,
    ) -> None:
        self.role, self.path, self.in_code, self.json_end = role, path, in_code, json_end
        self.extra_texts, self.before_end, self.first_cells = extra_texts, before_end, first_cells
        self.texts_override = texts_override
        self.texts: list[str] = []
        self.queue: Optional[list[dict]] = None
        self.planned = False
        self.cells: list[tuple[str, str]] = []
        self.terminal: list[str] = []
        self.namespace: set[str] = set()
        self.outputs: dict[str, str] = {}
        self.done = asyncio.Event()
        self.n = 0

    def _cell(self, code: str) -> dict:
        self.n += 1
        return _call(f"{self.role}-{self.n}", "execute_code", {"code": code})

    async def __call__(self, messages, tools):
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = str(m["content"])
        if self.queue is None:
            names = [t["function"]["name"] for t in tools]
            prompt = [str(m.get("content") or "") for m in messages if m.get("role") in ("system", "user")]
            self.texts = (
                list(self.texts_override)
                if self.texts_override is not None
                else [*prompt, *(t["function"]["description"] for t in tools), *self.extra_texts]
            )
            self.queue = [self._cell(c) for c in self.first_cells] + [self._cell(LIST_NAMES)]
            self.json_tools = frozenset(names) - {"execute_code"}
            return self.queue.pop(0), "0"
        if self.queue == [] and not self.planned:
            # the namespace is known: plan the shown calls against it
            self.planned = True
            listed = names_listed(self.outputs[f"{self.role}-{len(self.first_cells) + 1}"])
            self.namespace = listed
            body, last = plan(self.texts, listed | set(dir(builtins)), self.path, self.json_tools)
            self.cells = body
            self.queue = [self._cell(wrapped(i, code)) for i, (_, code) in enumerate(body)]
            self.terminal = [code for _, code in last]
            if not self.queue:
                self.queue = [self._cell("print('no shown calls')")]
        if self.queue:
            return self.queue.pop(0), "0"
        if self.before_end is not None:
            await self.before_end()
        self.done.set()
        if self.in_code:
            end = self.terminal[0] if self.terminal else self.json_end_code()
            self.terminal = []
            return self._cell(end), "0"
        self.n += 1
        return _call(f"{self.role}-{self.n}", *self.json_end), "0"

    def json_end_code(self) -> str:
        name, args = self.json_end
        return f"tools.{name}(**{args!r})"

    def report(self) -> list[str]:
        """Every shown call that broke the contract, with why."""
        outputs = "\n".join(self.outputs.values())
        got = verdicts(outputs)
        bad = []
        for i, (span, code) in enumerate(self.cells):
            why = failed(got.get(i))
            if why is not None:
                bad.append(f"{self.role}: `{span}` ran as {code!r}: {why}")
        for text in self.texts:
            for name in said_bound(text) - self.namespace:
                bad.append(f"{self.role}: the prompt says `{name}` is already available, but it is not bound")
        return bad



def _library_pass(tmp_path: Path, model: Any, *, in_code: bool, **cfg: Any):
    """A library-mode writer pass over episode ``e1`` whose parent commit holds the fixture library."""
    from unify.memory.blobs import BlobStore
    from unify.memory.evidence import EvidenceStore
    from unify.memory.gate import Gate
    from unify.memory.gitio import Repo
    from unify.memory.writer_pass import PassConfig, WriterPass

    mem = Repo.init_bare(tmp_path / "mem.git")
    _commit(mem, LIB, "seed the library")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    cfg.setdefault("max_calls", 400)
    cfg.setdefault("deadline_s", 900.0)
    return WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(
            library_mode=True,
            tools_in_code=in_code,
            pass_record_root=str(tmp_path / "out"),
            **cfg,
        ),
    )


@pytest.fixture
def no_waiting(monkeypatch):
    """``wait`` and ``ask`` answer at once (each stands for a call the host answers later)."""
    from unify.memory import tool_handlers

    async def wait(ctx, args):
        return "no new messages"

    async def ask(ctx, args):
        return "no answer yet"

    monkeypatch.setattr(tool_handlers, "wait", wait)
    monkeypatch.setattr(tool_handlers, "ask", ask, raising=False)


def _switch(monkeypatch, in_code: bool) -> None:
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on" if in_code else "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")


NOTE = "/memory/notes/text/dates.md"


@needs_box
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("in_code", [True, False], ids=["tools-in-code", "tools-off"])
def test_every_call_the_writers_and_a_helpers_prompts_show_runs_as_written(tmp_path, monkeypatch, no_waiting, in_code):
    from unify.memory.trigger import PassRequest

    _switch(monkeypatch, in_code)
    helpers_made: list[Contract] = []

    def helper_factory(label=None):
        if helpers_made:  # any other helper a shown delegate call starts ends at once
            quick = Contract("extra-helper", NOTE, in_code=in_code, json_end=("done", {"summary": "s", "claims": []}))
            quick.queue, quick.planned = [], True
            return quick
        helper = Contract("helper", NOTE, in_code=in_code, json_end=("done", {"summary": "s", "claims": []}))
        helpers_made.append(helper)
        return helper

    async def helper_ended():
        for _ in range(1200):
            if helpers_made and helpers_made[0].done.is_set():
                return
            await asyncio.sleep(0.5)

    start = (
        ("print(tools.delegate('Check the library index against e1.', ['e1']))",)
        if in_code
        else ()
    )
    writer_model = Contract(
        "writer",
        NOTE,
        in_code=in_code,
        json_end=("finish", {"summary": "contract test summary"}),
        extra_texts=(R.EXECUTE_CODE.description,) if in_code else (),
        before_end=helper_ended,
        first_cells=start,
    )
    if not in_code:
        # the helper is started by the model-API delegate call before the first cell
        original = writer_model.__call__

        async def first_delegate(messages, tools, _sent=[]):  # noqa: B006
            if not _sent:
                _sent.append(True)
                args = {"task": "Check the library index against e1.", "episodes": ["e1"]}
                return _call("writer-delegate", "delegate", args), "0"
            return await original(messages, tools)

        model = first_delegate
    else:
        model = writer_model
    writer = _library_pass(tmp_path, model, in_code=in_code)
    writer.helper_turn = helper_factory
    asyncio.run(writer.run(PassRequest("incremental", "text", ["e1"], False), "p1"))
    bad = writer_model.report()
    assert writer_model.cells, writer_model.texts[:1]
    assert helpers_made, "no helper was started"
    helper = helpers_made[0]
    # with tools in code off the helper's brief shows no Python call (its tools are model-API tools): only its
    # namespace is checked then
    assert helper.done.is_set() and (helper.cells or not in_code), helper.outputs
    bad += helper.report()
    assert not bad, "\n".join(bad)
    for agent, contract in (("writer", writer_model), ("helper", helper)):
        assert "memory" in contract.namespace and "help" in contract.namespace, (agent, contract.namespace)
        assert ("tools" in contract.namespace) is in_code, (agent, contract.namespace)
        # the reverse: every pre-bound name is named in the prompt
        joined = "\n".join(contract.texts)
        for name in R.prebound_names(agent, {"tools": in_code, "memory": True}):
            assert f"`{name}`" in joined, (agent, name)


@needs_box
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("in_code", [True, False], ids=["tools-in-code", "tools-off"])
def test_every_call_the_curators_prompt_shows_runs_as_written(tmp_path, monkeypatch, no_waiting, in_code):
    """The curator's brief, rendered by its builder, run in the curator's box (the writer pass's box: the same
    binds, profile and names); every tool it names is on the curator's allow-list."""
    from unify.memory import library_prompts as lp
    from unify.memory.trigger import PassRequest

    _switch(monkeypatch, in_code)
    brief = lp.curate_brief_now(edit_tools=True)
    texts = [brief, *((R.EXECUTE_CODE.description,) if in_code else ())]
    named = R.tool_names_in(brief) - {"list", "help", "Refused", "NotRun"}
    assert named <= set(R.allow_list("curator", lp.tool_flags(edit=True))), named
    model = Contract(
        "curator",
        NOTE,
        in_code=in_code,
        json_end=("finish", {"summary": "contract test summary"}),
        texts_override=texts,
    )
    writer = _library_pass(tmp_path, model, in_code=in_code, edit_tools=True)
    asyncio.run(writer.run(PassRequest("incremental", "text", ["e1"], False), "p1"))
    assert model.cells
    bad = model.report()
    assert not bad, "\n".join(bad)
    assert "memory" in model.namespace and ("tools" in model.namespace) is in_code


def test_the_fork_names_no_pre_bound_name():
    from unify.memory.integration import fork

    text = fork.fork_instruction()
    assert "Do not call" in text
    assert not said_bound(text) and "already bound" not in text


def test_every_listed_span_that_is_not_run_is_still_in_a_prompt(monkeypatch):
    from unify.actor import prompt_builders as pb
    from unify.actor.core_surface import PromptSurface
    from unify.memory import helpers
    from unify.memory import library_prompts as lp

    shown: set[str] = set()
    for switch in ("on", "off"):
        monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", switch)
        full = {"staging": True, "pool": True, "helper": True, "similar": True, "edit_tools": True}
        for text in (
            pb.build_code_act_prompt(
                environments={},
                core=PromptSurface(functions=False, guidance=False),
                session_sections=False,
            ),
            lp.write_brief_now(**full),
            lp.curate_brief_now(edit_tools=True),
            helpers.helper_brief(True),
        ):
            shown |= {s.strip() for s in shown_code(text)}
    for span in NOT_RUN:
        assert span in shown, span


# --- the actor and a teammate: the real request in the real sandbox ------------------------------------------------


def _actor_library(world: dict, monkeypatch) -> Any:
    """A library-mode memory run whose read-only copy holds the fixture library, current for the request."""
    from tests.memory.integration.test_actor_export_library_mode import _library_repo
    from unify.memory.integration import request as request_mod
    from unify.memory.integration.checkout import export_actor_library_mode
    from unify.memory.integration.paths import Paths
    from unify.memory.integration.prompt import render_memory_library_mode
    from unify.memory.integration.request import RequestRun

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    run = RequestRun("r", paths)
    run.library_mode = True
    run.index, run.shown = render_memory_library_mode(paths.checkout)
    run.memory_bound = True
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


def _actor_stubs(monkeypatch, names: tuple[str, ...]) -> None:
    """The actor's host handlers for *names* (calls that would install, wait or start and stop agents): fixed text."""
    from unify.actor import tool_handlers as H

    texts = {
        "install": "installed (stand-in)",
        "wait": "no new messages",
        "ask": "no answer yet",
        "delegate": "started (stand-in)",
        "stop": "stopped (stand-in)",
        "store_skills": "stored (stand-in)",
    }

    def fixed(text):
        async def handler(ctx, args):
            return H._reply(text)

        return handler

    for name in names:
        monkeypatch.setattr(H, name, fixed(texts[name]), raising=False)


def _actor_turns(seen: list):
    """The scripted turns of the agent under test: the namespace, then each shown call of the prompt it was sent,
    then the reply (which sets ``state["done"]``)."""
    from tests.scripted_model import reply

    def cell(code: str) -> tuple[str, dict]:
        return ("execute_code", {"code": code})

    state: dict = {"queue": None, "cells": [], "texts": [], "namespace": set(), "done": asyncio.Event()}

    def text_of(content) -> str:
        return content if isinstance(content, str) else json.dumps(content)

    def turn(call):
        seen.append("\n".join(text_of(m.get("content")) for m in call.messages if m.get("role") == "tool"))
        if state["queue"] is None:
            state["texts"] = [
                text_of(m.get("content")) for m in call.messages if m.get("role") in ("system", "user")
            ] + [t["function"]["description"] for t in call.request.get("tools") or []]
            state["queue"] = [LIST_NAMES]
            state["listing"] = True
        elif state.get("listing"):
            state["listing"] = False
            state["namespace"] = names_listed(seen[-1])
            body, _ = plan(state["texts"], state["namespace"] | set(dir(builtins)), "/workspace")
            state["cells"] = body
            state["queue"] = [wrapped(i, code) for i, (_, code) in enumerate(body)]
        if state["queue"]:
            return reply(calls=[cell(state["queue"].pop(0))])
        state["done"].set()
        return reply("contract test answer")

    return turn, state


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("agent", ["actor", "teammate"])
async def test_every_call_the_actors_prompt_shows_runs_as_written(core_world, monkeypatch, agent):  # noqa: F811
    """The actor's request (and a teammate's, started by the actor's own delegate call) in the real sandbox: its
    system message (the shared opening, the core prompt and the library guide with the index), its first message
    and its one tool's text. The calls that end the turn are the scripted reply."""
    from tests.actor.code_act.test_in_code_tools_wiring import _Collect
    from tests.scripted_model import Always, ScriptedModel, reply, scripted
    from unify.actor import in_code_tools
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(in_code_tools, "SESSIONS", _Collect())
    _actor_library(core_world, monkeypatch)
    seen: list[str] = []
    turn, state = _actor_turns(seen)
    if agent == "actor":
        _actor_stubs(monkeypatch, ("install", "wait", "ask", "delegate", "stop", "store_skills"))
        model = ScriptedModel(actor=[Always(turn)], query_llm=Always(reply("ok")))
    else:
        # the actor's own delegate call starts the teammate for real; the teammate's own shown delegate calls answer
        # fixed text (a second teammate would share the scripted turns); the actor answers once the teammate is done
        _actor_stubs(monkeypatch, ("install", "wait", "ask", "stop", "store_skills"))
        from unify.actor import tool_handlers as H

        real_delegate = H.delegate

        async def delegate(ctx, args):
            if getattr(ctx, "agent", None) == "teammate":
                return H._reply("started (stand-in)")
            return await real_delegate(ctx, args)

        monkeypatch.setattr(H, "delegate", delegate)

        async def answer_after_the_teammate(call):
            await asyncio.wait_for(state["done"].wait(), 600)
            return reply("contract test answer")

        model = ScriptedModel(
            actor=[
                reply(calls=[("execute_code", {"code": START_TEAMMATE})]),
                Always(answer_after_the_teammate),
            ],
            helper=[Always(turn)],
            query_llm=Always(reply("ok")),
        )
    actor = CodeActActor()
    try:
        with scripted(model):
            handle = await actor.act("Run every call your instructions show.", persist=False)
            await asyncio.wait_for(handle.result(), 700)
    finally:
        await actor.close()
    assert state["cells"], state["texts"][:1]
    got = verdicts("\n".join(seen))
    bad = [
        f"{agent}: `{span}` ran as {code!r}: {why}"
        for i, (span, code) in enumerate(state["cells"])
        if (why := failed(got.get(i))) is not None
    ]
    for text in state["texts"]:
        for name in said_bound(text) - state["namespace"]:
            bad.append(f"{agent}: the prompt says `{name}` is already available, but it is not bound")
    assert not bad, "\n".join(bad)
    for name in ("tools", "memory", "display", "help", "query_llm", "list_llms", "unillm", "run_coro_sync"):
        assert name in state["namespace"], (agent, name, sorted(state["namespace"]))
    joined = "\n".join(state["texts"])
    for name in R.prebound_names(agent, {"tools": True, "memory": True}):
        assert f"`{name}`" in joined, (agent, name)
