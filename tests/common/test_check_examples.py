"""The example check every code agent has: ``tools.check_examples(candidate, examples)``.

It runs a function the agent defined in its own session on each example's input, in the agent's own interpreter
(the ``tools`` module embeds it; it is never a host call), and reports per example the input, the expected and the
actual output (or the exception), whether it passed and the difference from the one comparison the gate uses
(:func:`unify.memory.value_diff.difference`, the module's own source, run in the box). The actor's and a teammate's
opening holds one line about it.

Keyless and boxless: the ``tools`` module is rendered from the registry with a stand-in channel client that records
every channel call, and loaded in the test's interpreter (the box tests run it in the real box).
"""

from __future__ import annotations

import inspect
import types

import pytest

from unify.common import box_check_source
from unify.common import tool_registry as R
from unify.memory import value_diff
from tests.memory.prompt_guards import tool_names_in

FULL = frozenset({"pool", "similar", "helper", "edit", "store", "examples"})
CODE_AGENTS = ("actor", "teammate", "writer", "helper")  # rev 2 has no curator role

#: A channel client that records every call (a check must make none).
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
    return Result("%s answered" % tool)
'''
SHELL = "class ShellResult(str):\n    pass\n\n\ndef sh(command, timeout=None):\n    return command\n"


@pytest.fixture
def load(monkeypatch):
    monkeypatch.setattr(R, "client_source", lambda: CLIENT)
    monkeypatch.setattr(R, "shell_source", lambda: SHELL)

    def _load(agent: str = "actor") -> types.ModuleType:
        mod = types.ModuleType("tools")
        exec(compile(R.render_stub(R.allow_list(agent, FULL), agent), "<tools>", "exec"), mod.__dict__)
        return mod

    return _load


def _calls(tools: types.ModuleType) -> list:
    return tools._client["CALLS"]


def totals(order):
    return {"count": len(order["prices"]), "sum": sum(order["prices"])}


# --- who has it, and where it runs ----------------------------------------------------------------------------------


def test_every_code_agent_has_it_and_it_never_reaches_the_host(load):
    from unify.actor import in_code_tools

    spec = R.registry()["check_examples"]
    assert spec.kind == "local" and not spec.handler
    for agent in CODE_AGENTS:
        assert "check_examples" in R.allow_list(agent, FULL), agent
        tools = load(agent)
        report = tools.check_examples(totals, [{"input": {"prices": [2, 3]}, "expected": {"count": 2, "sum": 5}}])
        assert report.ok and _calls(tools) == [], agent
    assert "check_examples" not in R.allow_list("librarian", FULL)
    # never on a channel's table: the stub runs it, as it runs sh
    allow = R.allow_list("actor", FULL)
    assert "check_examples" not in in_code_tools.channel_specs("actor", allow)
    # the writer's own check is still the gate's
    assert "check" in R.allow_list("writer", FULL) and R.registry()["check"].kind == "action"
    # rev 2: tools are always in code, so no agent has it as a model-API tool (that is execute_code alone)
    for agent in ("writer", "helper"):
        assert "check_examples" not in [t["function"]["name"] for t in R.json_tools(agent)], agent


def test_the_comparison_is_the_gates_own_function(load):
    tools = load()
    stub = R.render_stub(R.allow_list("actor", FULL), "actor")
    assert repr(inspect.getsource(value_diff)) in stub
    assert repr(box_check_source.SOURCE) in stub
    pairs = [({"a": [1, 2]}, {"a": [1, 3]}), ("one\ntwo\n", "one\nthree\n"), (42, "42"), ([1], {"x": 1})]
    for expected, actual in pairs:
        report = tools.check_examples(lambda _x, a=actual: a, [{"input": 0, "expected": expected}])
        assert report.results[0]["difference"] == value_diff.difference(expected, actual)
        assert report.results[0]["passed"] is value_diff.same(expected, actual)


# --- per example: passed, failed, raised -------------------------------------------------------------------------


def test_each_example_passes_fails_or_raises_with_its_values_json(load):
    tools = load()
    report = tools.check_examples(
        totals,
        [
            {"input": {"prices": [2, 3]}, "expected": {"count": 2, "sum": 5}},
            {"input": {"prices": [1]}, "expected": {"count": 1, "sum": 2}},
            {"input": {"cost": 1}, "expected": {"count": 0, "sum": 0}},
        ],
    )
    lines = report.splitlines()
    assert lines[0] == "passed 1 of 3"
    assert (report.passed, report.total, report.ok) == (1, 3, False)
    assert [r["passed"] for r in report.results] == [True, False, False]
    passed, failed, raised = report.results
    assert passed["actual"] == {"count": 2, "sum": 5} and passed["difference"] is None and passed["error"] is None
    assert failed["actual"] == {"count": 1, "sum": 1} and failed["error"] is None
    assert "sum: expected 2, got 1" in failed["difference"]
    assert raised["actual"] is None and raised["difference"] is None
    assert raised["error"].startswith("KeyError: 'prices' (line ") and raised["error"].endswith(", in totals)")
    # the text: the input of each, and expected, actual and the difference of the one that failed
    assert "example 1: passed" in lines and "  input: {'prices': [2, 3]}" in lines
    i = lines.index("example 2: failed")
    assert lines[i + 1 : i + 4] == [
        "  input: {'prices': [1]}",
        "  expected: {'count': 1, 'sum': 2}",
        "  actual: {'count': 1, 'sum': 1}",
    ]
    assert lines[i + 4] == "  difference:" and '    expected: {"count":1,"sum":2}' in lines
    assert '    actual:   {"count":1,"sum":1}' in lines and "    sum: expected 2, got 1" in lines
    j = [n for n, line in enumerate(lines) if line.startswith("example 3: raised KeyError: 'prices'")]
    assert j and lines[j[0] + 1 : j[0] + 3] == ["  input: {'cost': 1}", "  expected: {'count': 0, 'sum': 0}"]


def test_text_examples_get_a_unified_diff_that_names_expected_and_actual(load):
    tools = load("teammate")

    def shout(text):
        return text.upper()

    report = tools.check_examples(shout, [("a\nb\nc", "A\nB\nC"), ("x\ny", "X\nZ"), ("q", "  Q\n")])
    assert report.splitlines()[0] == "passed 2 of 3"
    diff = report.results[1]["difference"]
    assert diff.splitlines()[:2] == ['expected: "X\\nZ"', 'actual:   "X\\nY"']
    assert "--- expected" in diff and "+++ actual" in diff and "-Z" in diff and "+Y" in diff
    # text is compared without its leading and trailing whitespace
    assert report.results[2]["passed"]


def test_both_example_forms_and_an_output_key_are_taken(load):
    tools = load()

    def double(x):
        return 2 * x

    forms = [
        [{"input": 2, "expected": 4}, {"input": 3, "output": 6}],
        [(2, 4), [3, 6]],
        ((2, 4), (3, 6)),
        [{"input": 2, "expected": 4}, (3, 6)],
    ]
    for examples in forms:
        report = tools.check_examples(double, examples)
        assert report.ok and report.total == 2, examples
    # one example given alone
    assert tools.check_examples(double, {"input": 5, "expected": 10}).ok
    # any number of examples: nothing caps them
    many = tools.check_examples(double, [(n, 2 * n) for n in range(3000)])
    assert many.splitlines()[0] == "passed 3000 of 3000"
    empty = tools.check_examples(double, [])
    assert empty == "passed 0 of 0: no examples were given" and not empty.ok


def test_an_input_is_spread_over_several_positional_arguments_and_is_copied(load):
    tools = load()

    def add(a, b):
        return a + b

    def length(xs):
        return len(xs)

    assert tools.check_examples(add, [([1, 2], 3), ((2, 2), 4)]).ok
    # a function of one argument takes the list whole
    assert tools.check_examples(length, [([1, 2, 3], 3)]).ok

    def grow(xs):
        xs.append(0)
        return len(xs)

    examples = [([1], 2)]
    report = tools.check_examples(grow, examples)
    assert report.ok and examples == [([1], 2)] and report.results[0]["input"] == [1]


def test_long_values_are_cut_in_the_text_and_whole_in_the_results(load):
    tools = load()
    long = "x" * 5000
    report = tools.check_examples(lambda s: s, [(long, "y")])
    assert f"… [{len(repr(long)) - box_check_source.SHOWN_CHARS} more characters not shown]" in report
    assert long not in report and report.results[0]["input"] == long and report.results[0]["actual"] == long


# --- a malformed call teaches --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call, error, problem",
    [
        (lambda t: t.check_examples(3, [(1, 1)]), TypeError, "candidate is int, not a function"),
        (lambda t: t.check_examples(len, "abc"), TypeError, "examples is str, not a list of examples"),
        (lambda t: t.check_examples(len, [{"in": 1, "expected": 1}]), ValueError, "example 1 has the keys"),
        (lambda t: t.check_examples(len, [(1, 2, 3)]), ValueError, "example 1 is tuple, not a dict or a pair"),
        (lambda t: t.check_examples(len, [{"input": 1, "expected": 1, "output": 1}]), ValueError, "example 1 has"),
    ],
)
def test_a_malformed_call_names_the_right_call_with_an_example(load, call, error, problem):
    tools = load()
    with pytest.raises(error) as raised:
        call(tools)
    text = str(raised.value)
    assert text.startswith("tools.check_examples(") and problem in text, text
    assert "Expected tools.check_examples(candidate, examples) with a list of examples" in text, text
    assert 'e.g. tools.check_examples(solve, [{"input": [2, 3], "expected": 5}' in text, text


def test_a_wrong_number_of_arguments_teaches_as_every_tool_does(load):
    tools = load()
    with pytest.raises(TypeError) as raised:
        tools.check_examples(len)
    text = str(raised.value)
    assert "missing 1 required positional argument: 'examples'" in text, text
    assert "tools.check_examples(candidate, examples)" in text, text


# --- its help and the opening ---------------------------------------------------------------------------------------


def test_the_help_is_typed_and_shows_a_json_and_a_text_example(load, capsys):
    text = R.help_text("check_examples", "actor")
    assert text.startswith("check_examples(candidate, examples)\n")
    assert "examples (array, required)" in text and "candidate (any, required)" in text
    assert "tools.check_examples(totals, [{" in text and "tools.check_examples(initials, [[" in text
    assert tool_names_in(text) <= set(R.registry()) | {"help"}
    tools = load()
    assert tools.check_examples.__doc__ == text
    # help returns its text and prints nothing (every call returns text)
    assert tools.help("check_examples") == text and capsys.readouterr().out == ""
    # the examples in the help run as written
    for example in R.registry()["check_examples"].examples:
        scope = {"tools": tools}
        exec(example, scope)
        assert capsys.readouterr().out.startswith("passed ") or scope.get("report").ok
    assert "check_examples(candidate, examples) - " in R.list_text(R.allow_list("actor", FULL), "actor")


def test_the_actors_and_a_teammates_opening_hold_the_line_once():
    for agent in ("actor", "teammate"):
        opening = R.preamble(agent, {"environment": "Your box: /workspace is writable.", "flags": FULL})
        assert opening.count(R.EXAMPLES_LINE) == 1, agent
        # rev 2: the record's rules come in the first message (COMMON_SECTION), never in the opening
        assert R.COMMON_SECTION not in opening, agent
    for agent in ("writer", "helper", "librarian"):
        assert R.EXAMPLES_LINE not in R.preamble(agent, {"flags": FULL}), agent
    assert R.EXAMPLES_LINE == (
        "When the request or the environment gives examples, check a candidate against them with "
        "`tools.check_examples` before acting on it."
    )
