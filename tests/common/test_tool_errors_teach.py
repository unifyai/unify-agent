"""Errors from the ``tools.*`` call path teach.

Each refusal and error a cell can meet through the channel or the box stub says what was given, what was expected
and one call that works for the caller's role (or the right thing to do next), with no unfilled template field and
no path on the host. One bad call per form:

* the channel's argument check (``bad_arguments``): a wrong type, a value outside an enum, a const, a missing or an
  unknown argument (at the top and nested), a value that matches no allowed form, too few and too many items;
* an unknown tool (``unknown_tool``), which also names the nearest tools;
* a handler that fails inside the harness;
* the stub's own Python errors: a wrong keyword, a missing or an extra argument, a name the stub does not have;
* ``tools.help``'s refusal templates.

Keyless and boxless: the "box" is the stdlib client run in threads of the test, as in ``test_tool_channel.py``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import re
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.contract_lint import _HOST_PATH, _UNFILLED
from unify.common import tool_channel as tc
from unify.common import tool_registry as R
from unify.memory.writer_tools import Reply

CLIENT = Path(tc.__file__).with_name("tool_channel_client.py")
FULL = frozenset({"pool", "similar", "helper", "edit", "store"})
#: A template field left unfilled, with or without a conversion (``{path}``, ``{path!r}``).
UNFILLED = re.compile(_UNFILLED.pattern[:-2] + r"(?:![rsa])?\}")


def teaches(text: str, given: str, *, right_call: str | None = None) -> list[str]:
    """What *text* lacks of an error that teaches: *given* (what was given), then what was expected with a call
    that works, or *right_call* (what to do next); nothing unfilled and no host path."""
    gaps = []
    if given not in text:
        gaps.append(f"does not say what was given ({given!r})")
    if right_call is None:
        if "Expected " not in text or "e.g. tools." not in text:
            gaps.append("no 'Expected ..., e.g. tools.<call>'")
    elif right_call not in text:
        gaps.append(f"no right call ({right_call!r})")
    if UNFILLED.search(text):
        gaps.append(f"an unfilled field {UNFILLED.search(text).group(0)}")
    if _HOST_PATH.search(text):
        gaps.append("a host path")
    return gaps


# --- the channel --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    name: str
    parameters: dict
    handler: Any
    kind: str = "reader"
    blocking: bool = True
    deadline_s: float | None = None
    examples: tuple = ()


def _obj(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


def boom(ctx, args):
    raise RuntimeError("/home/someone/run/secret.json is unreadable")


SPECS = {
    "boom": Spec("boom", _obj({}), boom),
    "pick": Spec(
        "pick",
        _obj(
            {
                "mode": {"const": "fast"},
                "slice": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2},
            },
        ),
        lambda ctx, args: Reply("picked"),
        examples=('tools.pick(mode="fast", slice=[1, 2])',),
    ),
}


def _binding(agent: str) -> tc.CallContext:
    return tc.CallContext(
        agent=agent,
        name=agent,
        role="write",
        pass_id="p1",
        request_id=None,
        roots={},
        limits=None,
        counters=None,
        record=None,
        episodes=None,
        deadline=time.monotonic() + 60.0,
        state=None,
    )


def _client(channel: tc.Channel) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"box_tools_{id(channel)}_{time.monotonic_ns()}", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    req, rep = channel.box_fds()
    mod._STATE["fds"] = (os.dup(req), os.dup(rep))
    return mod


def _calls(agent: str, allow: tuple[str, ...], calls: list[tuple[str, dict]], specs: dict | None = None) -> list:
    """What a cell of *agent* meets for each call (the reply's text, or the exception)."""

    async def main() -> list:
        rows: list[dict] = []
        channel = tc.Channel.open(_binding(agent), allow, rows.append, specs=specs)
        box = _client(channel)
        channel.after_spawn()
        channel.cell_started(0)
        task = asyncio.ensure_future(channel.serve())
        out = []
        loop = asyncio.get_running_loop()
        for tool, args in calls:
            fut = loop.create_future()

            def go(tool=tool, args=args, fut=fut) -> None:
                try:
                    got: Any = box._call(tool, args)
                except Exception as exc:  # noqa: BLE001 - the test reads what the cell would see
                    got = exc
                loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(got))

            threading.Thread(target=go, daemon=True).start()
            out.append(await fut)
        await channel.close("ended", record=False)
        for fd in box._STATE["fds"]:
            try:
                os.close(fd)
            except OSError:
                pass
        task.cancel()
        return out

    return asyncio.run(main())


CHANNEL_CASES = [
    # (agent, tool, args, form, what the text must say was given)
    ("writer", "read", {"path": 3}, "bad_arguments", "path is 3 (int)"),
    ("writer", "read", {"path": "/inputs/a.md", "offset": "x"}, "bad_arguments", "offset is 'x' (str)"),
    ("writer", "pool", {"op": "lst"}, "bad_arguments", "op is 'lst' (str)"),
    ("writer", "read", {}, "bad_arguments", "path is missing"),
    ("writer", "read", {"path": "/inputs/a.md", "pth": "x"}, "bad_arguments", "there is no argument 'pth'"),
    ("writer", "read_episode", {"episode": "e1", "parts": [5]}, "bad_arguments", "parts[0] is 5 (int)"),
    (
        "helper",
        "done",
        {"summary": "s", "claims": [{"claim": "c", "sources": ["e1/cell:1"], "note": "x"}]},
        "bad_arguments",
        "there is no argument 'note' in claims[0]",
    ),
    ("actor", "wait", {"timeout": "soon"}, "bad_arguments", "timeout is 'soon' (str)"),
]


@pytest.mark.parametrize("agent, tool, args, form, given", CHANNEL_CASES)
def test_each_argument_refusal_teaches_with_a_call_that_works_for_the_role(agent, tool, args, form, given):
    allow = tuple(n for n in R.allow_list(agent, FULL) if R.registry()[n].kind != "local")
    (got,) = _calls(agent, allow, [(tool, args)])
    assert type(got).__name__ == "Refused" and got.form == form, got
    text = str(got)
    assert text.startswith(f"refused: tools.{tool}("), text
    assert teaches(text, given) == [], text
    example = text.rsplit("e.g. ", 1)[1].split("\n")[0].rstrip(".")
    assert example == R.role_example(tool, agent), text


def test_an_enum_value_names_the_nearest_allowed_one():
    allow = tuple(n for n in R.allow_list("writer", FULL) if R.registry()[n].kind != "local")
    (got,) = _calls("writer", allow, [("pool", {"op": "lst"})])
    assert "Nearest: 'list'" in str(got)


def test_a_const_and_the_list_bounds_teach_on_a_spec_of_their_own():
    got = _calls(
        "writer",
        ("pick",),
        [("pick", {"mode": "slow"}), ("pick", {"slice": []}), ("pick", {"slice": [1, 2, 3]})],
        specs=SPECS,
    )
    givens = ["mode is 'slow' (str)", "slice has 0 items", "slice has 3 items"]
    expected = ["mode: 'fast'", "slice: list of int of at least 1 items", "slice: list of int of at most 2 items"]
    for exc, given, want in zip(got, givens, expected):
        assert exc.form == "bad_arguments" and teaches(str(exc), given) == [], str(exc)
        assert want in str(exc) and 'e.g. tools.pick(mode="fast", slice=[1, 2])' in str(exc), str(exc)


def test_the_validator_keeps_its_texts():
    schema = _obj({"a": {"type": "integer"}, "b": {"enum": ["x", "y"]}}, ("a",))
    assert tc.validate(schema, {"a": "1"}) == "'a' must be integer"
    assert tc.validate(schema, {"a": 1, "b": "z"}) == "'b' must be one of x, y"
    assert tc.validate(schema, {}) == "arguments needs 'a'"
    assert tc.validate(schema, {"a": 1, "c": 2}) == "unknown argument 'c' in arguments"
    assert tc.validate(schema, {"a": 1}) is None


def test_an_unknown_tool_names_the_nearest_tools_and_the_whole_list():
    (got,) = _calls("writer", ("read", "grep", "overview"), [("reed", {"path": "/inputs/a.md"})])
    text = str(got)
    assert got.form == "unknown_tool"
    assert text.startswith("refused: unknown tool 'reed'; nearest: read") and "nearest: " in text, text
    assert text.endswith("; your tools: read, grep, overview"), text
    assert teaches(text, "unknown tool 'reed'", right_call="your tools: read, grep, overview") == []


def test_a_handler_that_fails_is_the_harness_fault_with_what_to_do_next_and_no_message():
    (got,) = _calls("writer", ("boom",), [("boom", {})], specs=SPECS)
    text = str(got)
    assert type(got).__name__ == "ToolError"
    assert text.startswith("error: tools.boom failed inside the harness (RuntimeError).")
    assert "arguments were valid" in text and "Try the call once more" in text
    assert "secret" not in text and teaches(text, "tools.boom", right_call="continue without it") == []


# --- the stub -----------------------------------------------------------------------------------------------------

#: A stand-in channel client: answers every call, and raises a TypeError of its own for ``grep`` (an error inside a
#: call, which the stub passes on as it is).
STAND_IN = '''
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


def _call(tool, args):
    if tool == "grep":
        raise TypeError("inside the call")
    return Result("%s answered %s" % (tool, sorted(args)))
'''


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(R, "client_source", lambda: STAND_IN)
    shell = "class ShellResult(str):\n    pass\n\n\ndef sh(command, timeout=None):\n    return command\n"
    monkeypatch.setattr(R, "shell_source", lambda: shell)

    def load(agent: str) -> types.ModuleType:
        mod = types.ModuleType("tools")
        exec(compile(R.render_stub(R.allow_list(agent, FULL), agent), "<tools>", "exec"), mod.__dict__)
        return mod

    return load


STUB_CASES = [
    # (agent, the call, what the text must say was given)
    ("writer", lambda t: t.read(file="/inputs/a.md"), "tools.read(file='/inputs/a.md'): got an unexpected keyword"),
    ("writer", lambda t: t.read(), "tools.read(): missing 1 required positional argument: 'path'"),
    ("writer", lambda t: t.read("/inputs/a.md", 0, 1), "tools.read('/inputs/a.md', 0, 1): takes from 1 to 2"),
    # rev 2: wait's timeout is optional, so the actor's arity case is one argument too many
    ("actor", lambda t: t.wait(120, 5), "tools.wait(120, 5): takes from 0 to 1 positional"),
    ("actor", lambda t: t.delegate("Read e3", ["e3"]), "tools.delegate('Read e3', ['e3']): takes 1 positional"),
]


@pytest.mark.parametrize("agent, call, given", STUB_CASES)
def test_a_wrong_keyword_or_arity_teaches_with_the_roles_own_signature(stub, agent, call, given):
    tools = stub(agent)
    with pytest.raises(TypeError) as err:
        call(tools)
    text = str(err.value)
    assert teaches(text, given) == [], text
    name = given.split("(")[0].split(".")[1]
    assert f"Expected tools.{R.signature_text(name, agent)}, e.g. {R.role_example(name, agent)}." in text, text
    assert err.value.__cause__ is None and err.value.__suppress_context__


def test_a_misspelt_keyword_names_the_nearest_argument(stub):
    with pytest.raises(TypeError) as err:
        stub("writer").read(pth="/inputs/a.md")
    assert "Nearest: path." in str(err.value)


def test_a_name_the_stub_lacks_names_the_nearest_and_where_to_look(stub):
    tools = stub("writer")
    with pytest.raises(AttributeError) as err:
        tools.reed("/inputs/a.md")
    text = str(err.value)
    assert text.startswith("tools.reed: tools has no attribute 'reed'") and "Nearest: read" in text, text
    assert _names_in(text, _others("writer")) == set(), text
    assert "Refused" not in text, text
    assert teaches(text, "tools.reed", right_call="tools.list() lists yours.") == []
    assert not hasattr(tools, "finish_now") and not hasattr(tools, "_private") and not hasattr(tools, "__wrapped__")


def _others(agent: str) -> set[str]:
    """The registered tools *agent* does not have (``tools.<name>`` functions of other roles)."""
    return set(R.registry()) - set(R.allow_list(agent, FULL)) - {R.EXECUTE} - R.CHANNEL_ONLY


def _names_in(text: str, names: set[str]) -> set[str]:
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text)}


def test_a_tool_of_another_role_is_unavailable_and_only_the_callers_tools_are_suggested(stub):
    for agent, name in (("actor", "finish"), ("writer", "answer"), ("helper", "pool"), ("librarian", "pool")):
        tools = stub(agent)
        assert name in _others(agent), (agent, name)
        with pytest.raises(AttributeError) as err:
            getattr(tools, name)
        text = str(err.value)
        assert text.startswith(f"tools.{name} isn't available in your role.\n"), text
        assert teaches(text, f"tools.{name}", right_call="tools.list() lists yours.") == []
        assert _names_in(text, _others(agent) - {name}) == set(), text


def test_no_other_roles_tool_name_is_in_the_stubs_dir_or_its_digests(stub):
    for agent in R.AGENTS:
        tools, others = stub(agent), _others(agent)
        assert others and not others & set(dir(tools)), agent
        digests = tools._OTHER_ROLES
        assert all(re.fullmatch(r"[0-9a-f]{16}", d) for d in digests) and not others & set(digests), agent
        assert sorted(digests) == sorted(R.role_digest(n) for n in others), agent
        assert all(tools._other_role(n) for n in others), agent
        assert not any(tools._other_role(n) for n in (*R.allow_list(agent, FULL), "reed", R.EXECUTE)), agent


def test_the_channel_says_another_roles_tool_is_unavailable_and_names_only_the_callers_tools():
    allow = tuple(n for n in R.allow_list("actor", FULL) if R.registry()[n].kind != "local")
    (got,) = _calls("actor", allow, [("finish", {"answer": "x"})])
    text = str(got)
    assert got.form == "unknown_tool" and text.startswith("refused: tools.finish isn't available in your role; "), text
    assert text.endswith(f"your tools: {', '.join(allow)}"), text
    assert _names_in(text, _others("actor") - {"finish"}) == set(), text


def test_the_stub_keeps_its_signatures_help_and_errors_raised_inside_a_call(stub):
    tools = stub("writer")
    assert str(inspect.signature(tools.read)) == "(path, offset=None)"
    assert tools.read.__name__ == "read" and tools.read.__doc__ == R.help_text("read", "writer")
    assert tools.read("/inputs/a.md") == "read answered ['path']"
    with pytest.raises(TypeError, match=r"^inside the call$"):
        tools.grep("x", "/inputs")


def test_the_teaching_code_reads_the_same_renderer_as_the_memory_namespace():
    from unify.common import contract_runtime as cr

    source = R.errors_source()
    for fn in (cr.render, cr.shown, cr.call_text, cr.nearest):
        assert inspect.getsource(fn) in source
    assert "import " not in source.replace("import difflib", "")


def test_the_embedded_renderer_runs_alone_as_the_box_compiles_it():
    """The stub compiles the copied functions with nothing else of contract_runtime: one error renders the same."""
    import __future__

    from unify.common import contract_runtime as cr

    alone: dict = {}
    exec(compile(R.errors_source(), "<tools: errors>", "exec", __future__.annotations.compiler_flag, True), alone)
    args, kwargs = ("x" * 80,), {"offset": "x"}
    assert alone["call_text"]("tools.read", args, kwargs) == cr.call_text("tools.read", args, kwargs)
    near = alone["nearest"]("reed", ["read", "grep", "write"])
    assert near == cr.nearest("reed", ["read", "grep", "write"])
    parts = ("tools.read()", "path is missing")
    extra = {"expected": "path: str", "example": 'tools.read("a")', "nearest": near}
    extra["hint"] = "tools.list() lists yours."
    assert alone["render"](*parts, **extra) == cr.render(*parts, **extra)


# --- help ---------------------------------------------------------------------------------------------------------


def test_help_prints_no_unfilled_refusal_template():
    for agent in R.AGENTS:
        for name in R.allow_list(agent, FULL):
            text = R.help_text(name, agent)
            assert not UNFILLED.search(text), (agent, name, UNFILLED.search(text).group(0))
    assert "  outside_roots: <path> is outside /inputs, /memory and /outputs" in R.help_text("read")


# --- the tool only the memory package calls ------------------------------------------------------------------------


def test_an_unknown_tool_never_names_the_tool_only_the_memory_package_calls():
    own = tuple(n for n in R.allow_list("actor", FULL) if R.registry()[n].kind != "local")
    allow = own + tuple(R.CHANNEL_ONLY)
    (got,) = _calls("actor", allow, [("current_requst", {})])
    text = str(got)
    assert got.form == "unknown_tool" and text.startswith("refused: unknown tool 'current_requst'; "), text
    assert "current_request" not in text, text
    assert text.endswith(f"your tools: {', '.join(n for n in allow if n not in R.CHANNEL_ONLY)}"), text


def test_the_actors_stub_says_it_has_no_attribute_for_the_tool_only_the_memory_package_calls(stub):
    tools = stub("actor")
    assert "current_request" not in dir(tools)
    with pytest.raises(AttributeError) as err:
        tools.current_request()
    text = str(err.value)
    assert text.startswith("tools.current_request: tools has no attribute 'current_request'"), text
    assert "isn't available in your role" not in text, text
