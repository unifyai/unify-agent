"""Every sandbox name the actor's prompt advertises resolves in the actor's real sandbox, in each mode.

The tau2 airline runs (memory on, the simulation's tools registered as ``primitives.service``) told the actor that each
method's docs come back from ``await functions.search(...)``; a memory run's sandbox has no ``functions``, and 96 of
99 such calls raised ``NameError``, in 97 of 200 simulations. The prompt-to-runtime contract
(``tests/memory/test_prompt_runtime_contract.py``) runs the actor's shown calls, but with no environment registered,
so the namespaces section never rendered there.

Here an environment is registered (namespaces, a global and a module, as a world registers them) and the actor's
first request is taken as sent, in each memory mode a run uses: memory on in library mode (tau2, CL-Bench, office and
ARC memory cells) and memory off (no memory of any kind). Each name it advertises is read, without being called, in
one cell of the actor's real sandbox:

* every dotted name in a code span (``tools.read``, ``functions.search``, ``primitives.weather``, ``memory.ask``),
  and every bare name a span calls (``display``, ``query_llm``, ``help``);
* in a bullet that opens with an object's name (``- `functions`: ...``), a bare call is that object's method
  (``search(query)`` is ``functions.search``);
* every name a sentence says is already bound (``... are already available``, ``already bound, no import needed:``,
  ``binds these sandbox globals:``).

Names a span binds itself, standard-library modules (a cell imports them) and placeholder spans that do not parse
(``primitives.<namespace>.<method>(...)``) are not sandbox names. The few spans that name something on purpose
unbound are in :data:`NOT_BOUND` with the reason, and each must still be in the prompt of some mode.

Keyless: the model is scripted, the cell runs in the real box (bubblewrap and prlimit required).
"""

from __future__ import annotations

import ast
import asyncio
import re
import sys
import warnings
from typing import Optional

import pytest

from tests.actor.code_act.core_world import (
    core_world,
    new_actor,
    world,
)  # noqa: F401 (fixtures)
from tests.actor.code_act.test_in_code_tools_wiring import _Collect, _tool_results, code
from tests.memory.test_prompt_runtime_contract import (
    _FENCE,
    _SPAN,
    binds,
    said_bound,
)
from tests.memory.test_tool_channel_box import needs_box
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.settings import SETTINGS

#: Spans that name something the sandbox does not bind, on purpose, with the reason.
NOT_BOUND: dict[str, str] = {}  # none: the actor's prompt no longer carries the library index's example call
#: A bullet that opens with an object's name: its bare calls are that object's methods.
_OWNER = re.compile(r"^(?:- )?`(\w+)`:")
#: Sentences that say names are bound, besides the contract's (:func:`said_bound`).
_BOUND_SENTENCES = (
    re.compile(r"already bound, no import needed: (.*?)\.(?:\s|$)", re.S),
    re.compile(r"binds these sandbox globals: (.*?)\.(?:\s|$)", re.S),
)


def _dotted(node: ast.AST) -> Optional[str]:
    """``a.b.c`` for an attribute chain on a name (or the name), else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def span_names(span: str, owner: Optional[str]) -> set[str]:
    """The sandbox names *span* advertises: its longest dotted names, and the bare names it calls (as *owner*'s
    methods inside an owner's bullet). None for a span that does not parse or is an annotation alone (an item id).
    """
    try:
        tree = ast.parse(span)
    except SyntaxError:
        return set()
    if all(isinstance(n, ast.AnnAssign) and n.value is None for n in tree.body):
        return set()
    local = binds(span)
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and id(n) not in inner:
            name = _dotted(n)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            name = f"{owner}.{n.func.id}" if owner else n.func.id
        else:
            continue
        if name is None:
            continue
        root = name.split(".")[0]
        if root in local or root in sys.stdlib_module_names:
            continue
        out.add(name)
    return out


def spans(block: str) -> list[str]:
    """The fenced blocks and inline code spans of *block*; an inline span the prompt's line filling wrapped is read
    whole (the block's whitespace collapsed first)."""
    rest = " ".join(_FENCE.sub("", block).split())
    return [*_FENCE.findall(block), *_SPAN.findall(rest)]


def advertised(texts: list[str]) -> dict[str, str]:
    """Every sandbox name *texts* advertise, with the first span (or sentence) that names it."""
    out: dict[str, str] = {}
    for text in texts:
        # a bullet ends at the next bullet or at a blank line
        blocks = [b for para in text.split("\n\n") for b in para.split("\n- ")]
        for block in blocks:
            m = _OWNER.match(block)
            owner = m.group(1) if m else None
            for span in spans(block):
                span = span.strip()
                if span in NOT_BOUND:
                    continue
                for name in span_names(span, owner):
                    out.setdefault(name, span)
        for name in said_bound(text):
            out.setdefault(name, "(said already available)")
        for pattern in _BOUND_SENTENCES:
            for m in pattern.finditer(text):
                for name in re.findall(r"`(\w+)", m.group(1)):
                    out.setdefault(name, m.group(0)[:80])
    return out


def resolve_cell(names: list[str]) -> str:
    """One cell that reads each name (no call) and prints how each read ended."""
    lines = []
    for i, name in enumerate(names):
        lines.append(
            f"try:\n    {name}\n    print('RESOLVED {i} ok')\n"
            f"except Exception as exc:\n    print('RESOLVED {i} raised', type(exc).__name__, repr(str(exc))[:200])\n",
        )
    return "".join(lines)


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return "" if content is None else str(content)


def test_the_rules_find_the_names_a_span_advertises():
    assert span_names("await functions.search(...)", None) == {"functions.search"}
    assert span_names('page = tools.read("/path")', None) == {"tools.read"}
    assert span_names("help(primitives.<namespace>.<method>)", None) == set()
    assert span_names("memory.text.dates:parse_date", None) == set()
    assert span_names("search(query)", "functions") == {"functions.search"}
    assert span_names("display(obj)", None) == {"display"}
    assert span_names("asyncio.run(...)", None) == set()
    assert span_names("x = 1; x.bit_length()", None) == set()
    found = advertised(
        [
            "Intro `tools.list()`.\n- `functions`: stored. `get(name)`, `run(name)`\n- `install(packages)`: x"
            "\n\nAfter the list `display(obj)`.\n\n- `guidance`: stored. `search(text)`",
            "Cells run with these globals already bound, no import needed: `tools` (a), `unillm` (b); more.",
            "The environment also binds these sandbox globals: `weather_client`.",
        ],
    )
    assert spans("a `x =\n  f(1)` b") == ["x = f(1)"]
    assert set(found) == {
        "tools.list",
        "functions.get",
        "functions.run",
        "install",
        "display",
        "guidance.search",
        "tools",
        "unillm",
        "weather_client",
    }


def _memory_library(world: dict, monkeypatch) -> None:
    """A library-mode memory run whose read-only copy holds the fixture library, exported as the runs export it:
    with the librarian on, so the package has ``memory.ask`` (the contract test's copy is exported without it).
    """
    from tests.memory.integration.test_actor_export_library_mode import _library_repo
    from unify.memory.integration import request as request_mod
    from unify.memory.integration.checkout import export_actor_library_mode
    from unify.memory.integration.paths import Paths
    from unify.memory.integration.prompt import render_memory_section
    from unify.memory.integration.request import RequestRun

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    run = RequestRun("r", paths)
    run.library_mode = True
    run.index, run.shown = render_memory_section(paths.checkout)
    run.memory_bound = True
    monkeypatch.setattr(request_mod, "_CURRENT", run)


#: The modes a run uses: memory on in library mode, and memory off.
MODES = ("memory-library", "memory-off")


@pytest.fixture
def registered_world():
    """An environment registered as a world registers one: namespaces, a global and a module."""
    from tests.function_manager.primitives.test_environment_namespaces import (
        weather_surface,
    )
    from unify.function_manager.primitives import register_environment
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    clear_environment_namespaces()
    register_environment(weather_surface(), source="tests:weather_surface")
    yield
    clear_environment_namespaces()


async def _first_request_names(
    core_world,
    monkeypatch,
    mode: str,
) -> tuple[list[str], dict[str, str], str]:  # noqa: F811
    """The texts of the actor's first request in *mode*, the names they advertise, and the cell's output."""
    from unify.actor import in_code_tools
    from unify.actor.environments import registered_environments

    monkeypatch.setattr(in_code_tools, "SESSIONS", _Collect())
    if mode == "memory-library":
        _memory_library(core_world, monkeypatch)
    else:
        monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    state: dict = {}

    def turn(call):
        if "names" not in state:
            state["texts"] = [
                _text(m.get("content"))
                for m in call.messages
                if m.get("role") in ("system", "user")
            ]
            state["names"] = advertised(state["texts"])
            return reply(calls=[code(resolve_cell(sorted(state["names"])))])
        state["output"] = _tool_results(call)
        return reply("done")

    model = ScriptedModel(actor=[Always(turn)], allow=("query_llm",))
    actor = new_actor(environments=registered_environments())
    with scripted(model):
        handle = await actor.act(
            "Read every name your instructions advertise.",
            persist=False,
        )
        await asyncio.wait_for(handle.result(), 300)
    return state["texts"], state["names"], state.get("output", "")


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode", MODES)
async def test_every_name_the_actors_prompt_advertises_resolves_in_its_sandbox(
    core_world,  # noqa: F811
    registered_world,
    monkeypatch,
    mode,
):
    texts, names, output = await _first_request_names(core_world, monkeypatch, mode)
    ordered = sorted(names)
    warnings.warn(f"{mode} advertises {ordered}", stacklevel=1)
    got = {
        int(m.group(1)): m.group(2)
        for m in re.finditer(r"^RESOLVED (\d+) (ok|raised.*)$", output, re.M)
    }
    bad = [
        f"{mode}: `{name}` (from {names[name]!r}) {got.get(i, 'never reported')}"
        for i, name in enumerate(ordered)
        if got.get(i) != "ok"
    ]
    assert not bad, "\n".join(bad) + "\n" + output[-1500:]
    # the registered environment is advertised and read in every mode
    assert "primitives.weather" in names and "weather_client" in names, sorted(names)
    joined = "\n".join(texts)
    assert not re.search(r"\bfunctions\.\w", joined), joined[:3000]


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_every_span_not_bound_on_purpose_is_still_in_a_prompt(
    core_world,
    registered_world,
    monkeypatch,
):  # noqa: F811
    texts, _, _ = await _first_request_names(core_world, monkeypatch, "memory-library")
    joined = "\n".join(texts)
    for span in NOT_BOUND:
        assert span in joined, span
