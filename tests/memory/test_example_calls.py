"""The host side of the call record: every line of the plugin's file is bounded and checked, and a record is
complete only when the plugin loaded, an end line arrived and nothing was dropped or unreadable.
"""

import json
import os
import shutil

import pytest

from unify.memory import example_calls as ec


def _call(**over):
    row = {
        "event": "call",
        "item": "memory.grid.fill:flood_fill",
        "node": "memory/grid/tests/test_fill.py::test_fills[0]",
        "test": "memory/grid/tests/test_fill.py::test_fills",
        "indices": {"case": 0},
        "ordinal": 0,
        "file": "memory/grid/tests/test_fill.py",
        "line": 14,
        "args": [
            {
                "name": "grid",
                "sha": "ab",
                "type": "list",
                "json": [[0, 1]],
                "cut": False,
                "path": None,
            },
        ],
        "outcome": "returned",
        "raised": None,
        "result_sha": "cd",
        "result_type": "list",
        "result_json": [[3, 1]],
        "opened": ["memory/grid/tests/data/grid1.json"],
    }
    row.update(over)
    return row


def _write(path, rows, extra=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows) + extra)


def test_a_full_record_parses_and_is_complete(tmp_path):
    p = tmp_path / "calls.jsonl"
    _write(
        p,
        [
            {"event": "start", "targets": ["x"], "unobservable": None},
            _call(),
            {"event": "end", "calls": 1, "dropped": 0, "unresolved": []},
        ],
    )
    rec = ec.read_calls(p)
    assert rec.complete and len(rec.calls) == 1
    c = rec.calls[0]
    assert c.key == ec.CallKey(
        "memory/grid/tests/test_fill.py::test_fills",
        (("case", 0),),
        0,
    )
    assert (
        c.args[0].name == "grid"
        and c.args[0].json == [[0, 1]]
        and c.opened == ("memory/grid/tests/data/grid1.json",)
    )


def test_incomplete_records_are_never_complete(tmp_path):
    p = tmp_path / "calls.jsonl"
    _write(p, [{"event": "start"}, _call()])  # no end line
    assert not ec.read_calls(p).complete
    _write(
        p,
        [{"event": "start"}, _call(outcome="maybe"), {"event": "end", "dropped": 0}],
    )
    rec = ec.read_calls(p)
    assert rec.unreadable == 1 and not rec.complete
    _write(p, [{"event": "start"}, {"event": "end", "dropped": 3}])
    assert ec.read_calls(p).dropped == 3 and not ec.read_calls(p).complete
    _write(p, [{"event": "start"}, {"event": "end", "dropped": 0}])
    assert not ec.read_calls(p, loaded_property=False).complete
    _write(
        p,
        [{"event": "start"}, {"event": "end", "dropped": 0}],
        extra=b"x" * (ec.MAX_LINE_BYTES + 1) + b"\n",
    )
    assert ec.read_calls(p).unreadable == 1


def test_a_missing_or_linked_record_is_not_loaded(tmp_path):
    assert not ec.read_calls(tmp_path / "none.jsonl").loaded
    real = tmp_path / "real.jsonl"
    _write(real, [{"event": "start"}, {"event": "end", "dropped": 0}])
    os.symlink(real, tmp_path / "link.jsonl")
    assert not ec.read_calls(tmp_path / "link.jsonl").loaded


def test_the_plugin_is_written_and_the_box_told_its_targets(tmp_path):
    ec.plugin_files(tmp_path)
    assert (tmp_path / f"{ec.CALLS_MODULE}.py").read_text() == ec.CALLS_SOURCE
    compile(ec.CALLS_SOURCE, "_memory_calls.py", "exec")
    env = ec.calls_env(["b", "a", "a"])
    assert json.loads(env[ec.TARGETS_ENV]) == ["a", "b"] and env[ec.OUT_ENV] == ec.CALLS_OUT
    assert ec.unresolved_kind("memory.g.f:x (partial)") == ("memory.g.f:x", "partial")
    assert ec.unresolved_kind("memory.g.f:x") == ("memory.g.f:x", "not observed")


def _plugin_module():
    """The plugin's source as a module of this process; its hooks are never registered here."""
    import types

    mod = types.ModuleType("_memory_calls_under_test")
    exec(compile(ec.CALLS_SOURCE, "_memory_calls.py", "exec"), mod.__dict__)
    return mod


def _digest(value):
    import hashlib

    from unify.memory import fixtures

    return hashlib.sha256(fixtures.canonical(value).encode("utf-8")).hexdigest()


class _Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_argument_digests_are_the_canonical_digests_of_their_normal_form():
    """Each kind of value is digested as the canonical JSON of its normal form, the digest the host computes."""
    import datetime
    import decimal
    import hashlib

    plug = _plugin_module()
    cases = [
        ({"b": [1, 2], "a": {"c": None}}, {"b": [1, 2], "a": {"c": None}}),
        ([[0, 1], [1, 1]], [[0, 1], [1, 1]]),
        ((1, (2, 3)), [1, [2, 3]]),
        ({3, 1, 2}, [1, 2, 3]),
        (frozenset({"b", "a"}), ["a", "b"]),
        (b"ab", {"__bytes__": hashlib.sha256(b"ab").hexdigest()}),
        (decimal.Decimal("1.50"), "1.50"),
        (datetime.date(2026, 1, 2), "2026-01-02"),
        (datetime.datetime(2026, 1, 2, 3, 4), "2026-01-02T03:04:00"),
        (_Point(1, [2]), {"__type__": "_Point", "vars": {"x": 1, "y": [2]}}),
        ({1: "x"}, {"1": "x"}),
    ]
    for value, normal in cases:
        rec = plug.describe("v", value)
        assert rec["sha"] == _digest(normal), value
        assert rec["json"] == normal and rec["cut"] is False
        assert rec["type"] == type(value).__name__
    opaque = plug.describe("v", object())
    assert opaque["sha"] is None and opaque["json"] is None and opaque["type"] == "object"
    cyclic = []
    cyclic.append(cyclic)
    assert plug.describe("v", cyclic)["sha"] is None
    big = list(range(20000))
    rec = plug.describe("v", big)
    assert rec["cut"] is True and rec["json"] is None and rec["sha"] == _digest(big)


def test_outcomes_and_test_side_frames_are_classified_by_name_and_place():
    plug = _plugin_module()

    class MemoryInputError(ValueError):
        pass

    class Narrower(MemoryInputError):
        pass

    class Custom(Exception):
        pass

    assert plug.outcome_of(Narrower("x")) == ("refused", None)
    assert plug.outcome_of(KeyError("x")) == ("raised", "KeyError")
    assert plug.outcome_of(Custom("x")) == ("raised", "another exception")
    assert plug.tree_kind("/memory/memory/grid/tests/test_fill.py") == "test"
    assert plug.tree_kind("/memory/memory/grid/tests/helpers/load.py") == "test"
    assert plug.tree_kind("/memory/memory/grid/fill.py") == "library"
    assert plug.tree_kind("/memory/conftest.py") == "other"
    assert plug.tree_kind("<string>") == "other"
    assert plug.data_path("/memory/memory/grid/tests/data/g.json") == "memory/grid/tests/data/g.json"
    assert plug.data_path("/memory/memory/grid/tests/test_fill.py") is None
    assert plug.data_path("/memory/memory/grid/data/g.json") is None


# --- the plugin in the gate's box (skipped without bubblewrap, as every jail test) ---------------------------------

_LIB = '''
import functools


class MemoryInputError(ValueError):
    pass


class Narrower(MemoryInputError):
    pass


class Custom(Exception):
    pass


def flood_fill(grid, start, colour):
    out = [list(r) for r in grid]
    out[start[0]][start[1]] = colour
    return out


def outer(grid):
    return flood_fill(grid, (0, 0), 1)


def take(*args, **kwargs):
    return None


def refuse(x):
    raise Narrower("shape")


def lookup(k):
    return {"a": 1}[k]


def custom(x):
    raise Custom("x")


def gen(n):
    yield from range(n)


async def coro(x):
    return x


def _logged(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return fn(*a, **k)

    return wrapper


@_logged
def wrapped(x):
    return x + 1


not_a_function = functools.partial(flood_fill, colour=1)
'''

_HELPERS = '''
from memory.grid import fill


def call_fill(grid):
    return fill.flood_fill(grid, (1, 1), 7)
'''

_TESTS = '''
import asyncio
import datetime
import decimal
import json
import pathlib

import pytest

from memory.grid import fill
from memory.grid.tests import helpers

DATA = pathlib.Path(__file__).parent / "data"
GRID = json.loads((DATA / "grid1.json").read_text())
COLOURS = json.loads((DATA / "colours.json").read_text())


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


@pytest.mark.parametrize("colour", COLOURS)
def test_direct(colour):
    fill.flood_fill(GRID, (0, 0), colour)
    fill.flood_fill(GRID, (0, 1), colour)


def test_helper():
    helpers.call_fill(GRID)


def test_exec():
    exec("fill.flood_fill(g, (0, 0), 5)", {"fill": fill, "g": GRID})


def test_nested():
    fill.outer(GRID)


def test_wrapped():
    assert fill.wrapped(1) == 2


def test_arguments():
    fill.take(
        {"b": [1, 2]}, (1, 2), {3, 1, 2}, b"ab", decimal.Decimal("1.50"), datetime.date(2026, 1, 2),
        Point(1, 2), list(range(20000)), "memory/grid/tests/data/grid1.json", key=GRID,
    )


def test_outcomes():
    with pytest.raises(fill.MemoryInputError):
        fill.refuse(1)
    with pytest.raises(KeyError):
        fill.lookup("b")
    with pytest.raises(Exception):
        fill.custom(1)
    assert list(fill.gen(2)) == [0, 1]
    assert asyncio.run(fill.coro(3)) == 3
'''

_TARGETS = [
    f"memory.grid.fill:{n}"
    for n in ("flood_fill", "outer", "take", "refuse", "lookup", "custom", "gen", "coro", "wrapped", "not_a_function")
]

_jail = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")


def _tree(root, colours=(3, 4), tests=_TESTS):
    files = {
        "memory/grid/fill.py": _LIB,
        "memory/grid/tests/helpers.py": _HELPERS,
        "memory/grid/tests/test_fill.py": tests,
        "memory/grid/tests/data/grid1.json": json.dumps([[0, 1], [1, 1]]),
        "memory/grid/tests/data/colours.json": json.dumps(list(colours)),
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _run(tree, targets=_TARGETS):
    from unify.memory.sandbox_run import PYTHON, run_pytest

    return run_pytest(
        "/memory/memory/grid/tests",
        python=PYTHON,
        ro={tree: "/memory"},
        rw={},
        cwd="/memory",
        timeout_s=120,
        env={"PYTHONPATH": "/memory", "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib"},
        calls=targets,
    )


def _calls(out, item, test):
    return [c for c in out.calls.calls if c.item == f"memory.grid.fill:{item}" and c.key.test.endswith(test)]


@_jail
def test_the_plugin_records_calls_from_test_code_and_only_those(tmp_path):
    """A direct call, a call through a test helper under tests/, a call through exec in a test and a call through a
    library decorator are recorded; a library function's own call of a target is not. Call keys do not change when
    parameter values (and so pytest's ids) do."""
    out = _run(_tree(tmp_path / "a"))
    assert out.valid and not out.failed, out.output
    rec = out.calls
    assert rec.complete, (rec, out.output)
    assert rec.unresolved == ["memory.grid.fill:not_a_function (partial)"]
    direct = _calls(out, "flood_fill", "::test_direct")
    assert sorted((c.key.indices, c.key.ordinal) for c in direct) == [
        ((("colour", 0),), 0),
        ((("colour", 0),), 1),
        ((("colour", 1),), 0),
        ((("colour", 1),), 1),
    ]
    assert all(c.file == "memory/grid/tests/test_fill.py" and c.outcome == "returned" for c in direct)
    assert {c.node for c in direct} == {
        "memory/grid/tests/test_fill.py::test_direct[3]",
        "memory/grid/tests/test_fill.py::test_direct[4]",
    }
    first = min(direct, key=lambda c: c.key)
    assert [a.name for a in first.args] == ["grid", "start", "colour"]
    assert first.args[0].json == [[0, 1], [1, 1]] and first.args[2].json == 3
    assert first.result_json == [[3, 1], [1, 1]] and first.result_type == "list"
    assert "memory/grid/tests/data/grid1.json" in first.opened
    (helper,) = _calls(out, "flood_fill", "::test_helper")
    assert helper.file == "memory/grid/tests/helpers.py"
    (via_exec,) = _calls(out, "flood_fill", "::test_exec")
    assert via_exec.file == "memory/grid/tests/test_fill.py" and via_exec.args[2].json == 5
    assert not _calls(out, "flood_fill", "::test_nested")
    assert len(_calls(out, "outer", "::test_nested")) == 1
    assert [c.result_json for c in _calls(out, "wrapped", "::test_wrapped")] == [2]
    other = _run(_tree(tmp_path / "b", colours=(8, 9)))
    keys = sorted(c.key for c in _calls(other, "flood_fill", "::test_direct"))
    assert keys == sorted(c.key for c in direct)
    assert {c.node for c in _calls(other, "flood_fill", "::test_direct")} != {c.node for c in direct}


@_jail
def test_the_plugin_digests_arguments_as_the_host_does(tmp_path):
    import datetime
    import decimal
    import hashlib

    out = _run(_tree(tmp_path))
    (call,) = _calls(out, "take", "::test_arguments")
    args = {a.name: a for a in call.args}
    expected = {
        "args[0]": {"b": [1, 2]},
        "args[1]": [1, 2],
        "args[2]": [1, 2, 3],
        "args[3]": {"__bytes__": hashlib.sha256(b"ab").hexdigest()},
        "args[4]": str(decimal.Decimal("1.50")),
        "args[5]": datetime.date(2026, 1, 2).isoformat(),
        "args[6]": {"__type__": "Point", "vars": {"x": 1, "y": 2}},
        "kwargs.key": [[0, 1], [1, 1]],
    }
    for name, normal in expected.items():
        assert args[name].sha == _digest(normal), name
        assert args[name].json == normal and not args[name].cut
    big = args["args[7]"]
    assert big.cut and big.json is None and big.sha == _digest(list(range(20000)))
    assert args["args[8]"].path == "memory/grid/tests/data/grid1.json"
    assert args["args[0]"].path is None and args["args[2]"].type == "set"


@_jail
def test_the_plugin_records_how_each_call_ended(tmp_path):
    out = _run(_tree(tmp_path))
    by = {c.item.split(":")[1]: c for c in out.calls.calls if c.key.test.endswith("::test_outcomes")}
    assert (by["refuse"].outcome, by["refuse"].raised) == ("refused", None)
    assert (by["lookup"].outcome, by["lookup"].raised) == ("raised", "KeyError")
    assert (by["custom"].outcome, by["custom"].raised) == ("raised", "another exception")
    assert by["gen"].outcome == "returned" and by["gen"].result_sha is None
    assert by["coro"].outcome == "returned" and by["coro"].result_sha is None
    assert all(c.result_sha is None for c in (by["refuse"], by["lookup"], by["custom"]))


@_jail
def test_a_run_without_the_plugins_report_property_is_invalid_and_a_cut_record_incomplete(tmp_path):
    forged = (
        '<testsuites><testsuite><testcase classname="memory.grid.tests.test_fill" name="test_ok"/>'
        "</testsuite></testsuites>"
    )
    rewrites = (
        "import atexit\n"
        f"atexit.register(lambda: open('/junit/r.xml', 'w').write({forged!r}))\n"
        "from memory.grid import fill\n\n"
        "def test_ok():\n    fill.flood_fill([[0]], (0, 0), 1)\n"
    )
    out = _run(_tree(tmp_path / "a", tests=rewrites), targets=["memory.grid.fill:flood_fill"])
    assert not out.valid and out.calls is not None and not out.calls.loaded and not out.calls.complete
    cuts = (
        "import atexit\n"
        "def _cut():\n"
        "    lines = open('/calls/calls.jsonl').read().splitlines(True)\n"
        "    open('/calls/calls.jsonl', 'w').write(''.join(lines[:-1]))\n"
        "from memory.grid import fill\n\n"
        "def test_ok():\n    fill.flood_fill([[0]], (0, 0), 1)\n"
        "    atexit.register(_cut)\n"
    )
    out = _run(_tree(tmp_path / "b", tests=cuts), targets=["memory.grid.fill:flood_fill"])
    assert out.valid and out.calls.loaded and not out.calls.complete, out.output
