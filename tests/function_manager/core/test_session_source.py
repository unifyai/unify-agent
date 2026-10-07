"""Symbolic: ``UNIFY_STORE_FROM_SESSION``, the storage review stores a function the session ran by its name.

The review retyped code the session had run into the JSON string argument of
``add_functions``; on the 6-7 Oct office runs 16 of 41 retyped sources with
an escape gained an escaping level (``"\\n"`` for ``"\n"``), and the stored
function wrote a literal backslash-n on every reuse. With the switch on, an
implementation that is only a name is the source exactly as the session's
cell ran it. The trajectories are hand-built; nothing leaves the process.
"""

from __future__ import annotations

import json

import pytest

from unify.function_manager import session_source as ss
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

# A cell as the session ran it: the newline escape is one level, as Python wants it.
CELL = (
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "def write_names(names, path):\n"
    '    """Write one name per line."""\n'
    '    Path(path).write_text("\\n".join(names) + "\\n")\n'
    "    return json.dumps(names)\n"
    "\n"
    "print(write_names(['Ann', 'Bo'], 'out.txt'))\n"
)
NEWER = "def write_names(names, path):\n" '    return ";".join(names)\n'


def _call(call_id, code):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code, "language": "python"}),
                },
            },
        ],
    }


def _result(call_id, output="--- stdout ---\nok"):
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _trajectory(*cells):
    out = [{"role": "user", "content": "Write the names, one per line."}]
    for i, code in enumerate(cells):
        out += [_call(f"c{i}", code), _result(f"c{i}")]
    return out


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_FROM_SESSION", True)


def _stored(fm, name):
    return next(r for r in fm._library_rows() if r["name"] == name)["implementation"]


def test_the_switch_is_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_STORE_FROM_SESSION"].default is False


def test_a_name_stores_the_source_as_the_cell_ran_it(on):
    fm = FunctionManager(include_primitives=False)
    with ss.reviewing(_trajectory("print('looking')", CELL)):
        result = fm.add_functions(implementations=["write_names"])
    assert result == {"write_names": "added"}
    source = _stored(fm, "write_names")
    # One escaping level, exactly as in the cell: never "\\\\n".
    assert '"\\n".join(names) + "\\n"' in source
    assert "\\\\n" not in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["write_names"].__doc__ == "Write one name per line."


def test_the_imports_it_uses_move_into_its_body_after_the_docstring(on):
    fm = FunctionManager(include_primitives=False)
    with ss.reviewing(_trajectory(CELL)):
        fm.add_functions(implementations=["write_names"])
    lines = _stored(fm, "write_names").splitlines()
    assert lines[0] == "def write_names(names, path):"
    assert lines[1] == '    """Write one name per line."""'
    assert lines[2:4] == ["    from pathlib import Path", "    import json"]


def test_a_parameter_named_like_a_module_is_not_shadowed(on):
    cell = "import json\n\ndef dump(json):\n    return json.upper()\n"
    fm = FunctionManager(include_primitives=False)
    with ss.reviewing(_trajectory(cell)):
        fm.add_functions(implementations=["dump"])
    assert "import json" not in _stored(fm, "dump")


def test_the_latest_definition_wins(on):
    fm = FunctionManager(include_primitives=False)
    with ss.reviewing(_trajectory(CELL, NEWER)):
        fm.add_functions(implementations=["write_names"])
    assert _stored(fm, "write_names") == NEWER


def test_a_name_no_cell_defines_is_that_entrys_error_and_others_still_store(on):
    fm = FunctionManager(include_primitives=False)
    written = "def double(x):\n    return 2 * x\n"
    with ss.reviewing(_trajectory(CELL)):
        result = fm.add_functions(
            implementations=["missing_fn", written],
            raise_on_error=False,
        )
    assert result["double"] == "added"
    assert result["missing_fn"].startswith("error: no Python code cell")


def test_a_cell_that_did_not_run_or_is_not_python_is_not_used(on):
    traj = _trajectory("%%bash\necho hi\n")
    traj.append(_call("late", NEWER))  # no result: it never ran
    with ss.reviewing(traj):
        assert ss.resolve("write_names")[0] is None


def test_off_or_outside_a_review_nothing_is_resolved(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_FROM_SESSION", False)
    with ss.reviewing(_trajectory(CELL)):
        assert ss.expand(["write_names"]) == (["write_names"], {})
        assert ss.cells() == []
        assert ss.review_note() == ""
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_FROM_SESSION", True)
    assert ss.expand(["write_names"]) == (["write_names"], {})


def test_off_a_bare_name_is_refused_as_before(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_FROM_SESSION", False)
    fm = FunctionManager(include_primitives=False)
    with ss.reviewing(_trajectory(CELL)):
        result = fm.add_functions(
            implementations=["write_names"],
            raise_on_error=False,
        )
    assert result["implementation_1"].startswith("error:")
